import torch,os
import torchvision
import torch.nn as nn
import open3d as o3d
import numpy as np
import lightning as L
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding
from roma import rotmat_to_unitquat, quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from plyfile import PlyData, PlyElement

from utils.graphics_utils import compute_face_orientation
from utils.general_utils import inverse_sigmoid
from ..modules.ehm import EHM 
from ..modules.net_module.dino_encoder import  DINO_Enocder
from ..modules.net_module.feature_decoder import Vertex_GS_Decoder,UV_Point_GS_Decoder
from ..modules.net_module.styleunet import StyleUNet
from utils.graphics_utils import BaseMeshRenderer

#infer ubody gaussians from image and smplx
class Ubody_Gaussian_inferer(L.LightningModule):
    def __init__(self, cfg):
        super( ).__init__()
        self.cfg=cfg
        self.uvmap_size=cfg.uvmap_size
        
        n_harmonic_dir = 4
        self.direnc_dim = n_harmonic_dir * 2 * 3 + 3
        self.harmo_encoder = HarmonicEmbedding(n_harmonic_dir)
        self.intri_cam={'focal':cfg.invtanfov,'size': [cfg.image_size, cfg.image_size]}
        xy_image_coord=(get_pixel_coordinates(cfg.image_size,cfg.image_size)-0.5*cfg.image_size)/(cfg.image_size*0.5)# [-1,1]
        self.xy_image_coord=nn.Parameter(xy_image_coord, requires_grad=False)
        #vertex feature dim
        sample_out_dim=cfg.prj_out_dim 
        
        #general feature extrator
        self.dino_encoder=DINO_Enocder(output_dim=cfg.dino_out_dim,output_dim_2=sample_out_dim,hidden_dims=sample_out_dim//2)
        for param in self.dino_encoder.dino_model.parameters():
            param.requires_grad = False
        
        self.global_feature_mapping=nn.Sequential(nn.Linear(768,cfg.global_vertex_dim),nn.LeakyReLU(inplace=True),
                                                  nn.Linear(cfg.global_vertex_dim,cfg.global_vertex_dim),nn.LeakyReLU(inplace=True),
                                                  nn.Linear(cfg.global_vertex_dim,cfg.global_vertex_dim))
        self.vertex_gs_decoder = Vertex_GS_Decoder(in_dim=sample_out_dim+cfg.smplx_fea_dim+cfg.global_vertex_dim,
                                                   dir_dim=self.direnc_dim,color_out_dim=cfg.color_dim,)
        
        self.ehm=EHM( cfg.flame_assets_dir, cfg.smplx_assets_dir,add_teeth=cfg.add_teeth,uv_size=self.uvmap_size)
        self.smplx=self.ehm.smplx
        self.v_template=torch.nn.Parameter(self.ehm.v_template,requires_grad=False)
        self.laplacian_matrix=torch.nn.Parameter(self.ehm.laplacian_matrix,requires_grad=False)
        
        num_vertices=self.smplx.v_template.shape[0]
        self.vertex_base_feature = nn.Parameter(torch.randn(num_vertices, cfg.smplx_fea_dim), requires_grad=True,)
        
        #uv point feature decoder
        self.uv_feature_decoder = StyleUNet(in_size=self.uvmap_size, out_size=self.uvmap_size,activation=False,in_dim=cfg.dino_out_dim+3, out_dim=cfg.uv_out_dim,extra_style_dim=512)
        self.uv_style_mapping=nn.Sequential(nn.Linear(768,512),nn.LeakyReLU(inplace=True),nn.Linear(512,512),nn.LeakyReLU(inplace=True),nn.Linear(512,512))
        
        self.uv_base_feature = nn.Parameter(torch.randn((32,self.uvmap_size,self.uvmap_size)), requires_grad=True,)
        self.uv_point_decoder = UV_Point_GS_Decoder(in_dim=cfg.uv_out_dim+32, dir_dim=self.direnc_dim,color_out_dim=cfg.color_dim)#.to(device)
        self.mesh_renderer=BaseMeshRenderer(faces=self.smplx.faces_tensor,image_size=512,faces_uvs=self.smplx.faces_uv_idx,
                                       verts_uvs=self.smplx.texcoords,lbs_weights=self.smplx.lbs_weights,focal_length=self.cfg.invtanfov)
        self.uv_mask_flat=self.smplx.uvmap_mask.flatten()
           
    def sample_uv_feature(self,uv_coord,uv_feature_map):
        #uv_feature_map b c h w
        batch_size=uv_feature_map.shape[0]
        grid = uv_coord.clone()
        grid[..., 0] = 2.0 * uv_coord[..., 0] - 1.0  # u
        grid[..., 1] = 2.0 * uv_coord[..., 1] - 1.0  # v
        grid = grid[None,None].expand(batch_size,-1,-1,-1) #b 1 n 2
        sampled_features = nn.functional.grid_sample(uv_feature_map, grid, mode='bilinear', padding_mode='border', align_corners=False)
        sampled_features = sampled_features.squeeze(-2).permute(0,2,1).contiguous() #b n c
        return sampled_features
    
    def sample_prj_feature(self,vertices,feature_map,w2c_cam,vertices_img=None):
        batch_size=feature_map.shape[0]
        if vertices_img is None:
            vertices_homo=torch.cat([vertices,torch.ones_like(vertices[:,:,:1])],dim=-1)
            vertices_cam=torch.einsum('bij,bnj->bni',w2c_cam,vertices_homo)[:,:,:3]
            vertices_img=vertices_cam*self.cfg.invtanfov/(vertices_cam[:,:,2:]+1e-7)
        sampled_features = nn.functional.grid_sample(feature_map, vertices_img[:,None,:,:2], mode='bilinear', padding_mode='border', align_corners=False)
        sampled_features = sampled_features.squeeze(-2).permute(0,2,1).contiguous() #b n c
        return sampled_features,vertices_img
    
    def convert_pixel_feature_to_uv(self,img_features,deformed_vertices,w2c_cam,visble_faces=None,uv_features=None):
        batch_size,feature_dim=img_features.shape[0],img_features.shape[1]
        if uv_features is None:
            uv_features=torch.zeros((batch_size,feature_dim,self.cfg.uvmap_size,self.cfg.uvmap_size),device=img_features.device,dtype=torch.float32)
        uvmap_f_idx=self.smplx.uvmap_f_idx
        uvmap_f_mask=self.smplx.uvmap_mask
        uvmap_f_bary=self.smplx.uvmap_f_bary[None].expand(batch_size,-1,-1,-1)
        faces=self.smplx.faces_tensor
        uv_vertex_id=faces[uvmap_f_idx] #H W k
        uv_vertex=deformed_vertices.permute(1,2,0).contiguous()[uv_vertex_id]# H W k 3 B
        uv_vertex=uv_vertex.permute(4,0,1,2,3).contiguous()# B H W k 3
        uv_vertex= torch.einsum('bhwk,bhwkn->bhwn',uvmap_f_bary,uv_vertex)# B H W 3
        uv_vertex_homo=torch.cat([uv_vertex,torch.ones_like(uv_vertex[:,:,:,:1])],dim=-1)
        uv_vertex_cam=torch.einsum('bij,bhwj->bhwi',w2c_cam,uv_vertex_homo)[:,:,:,:3]
        vertices_img=uv_vertex_cam*self.cfg.invtanfov/(uv_vertex_cam[...,2:]+1e-7)
        uv_features = nn.functional.grid_sample(img_features, vertices_img[:,:,:,:2], mode='bilinear', padding_mode='zeros', align_corners=False)
        mask=self.smplx.uvmap_mask.clone()[None].repeat(batch_size,1,1)         
        if visble_faces is not None:
            num_faces=self.smplx.faces_tensor.shape[0]
            f_offset=torch.arange(batch_size,device=uvmap_f_idx.device,dtype=torch.int32)*num_faces

            all_faces=torch.arange(0,faces.shape[0],device=uvmap_f_idx.device,dtype=torch.int32)
            all_faces=all_faces[None].repeat(batch_size,1)
            all_faces=all_faces+f_offset[:,None]
            visble_all_faces=torch.isin(all_faces,torch.unique(visble_faces))
            visble_mask=visble_all_faces[:,uvmap_f_idx]
            mask=mask*visble_mask
            
        uv_features=uv_features*mask[:,None]
        return uv_features
        
    def forward(self,batch):
        batch_size = batch['image'].shape[0]
        extra_dict={}
        
        dino_feature_dict=self.dino_encoder(batch['image'],output_size=self.cfg.image_size)
        img_feature, img_feature_2,global_faeature=dino_feature_dict['f_map1'],dino_feature_dict['f_map2'],dino_feature_dict['f_global']
        vertex_global_feature=self.global_feature_mapping(global_faeature)
        
        cam_dirs=get_cam_dirs(batch["w2c_cam"])
        cam_dirs=self.harmo_encoder(cam_dirs)
        vertex_base_feature=self.vertex_base_feature[None].expand(batch_size,-1,-1)
        
        self.smplx_deform_res=self.ehm(batch['smplx_coeffs'],batch['flame_coeffs'])
        vertex_sample_feature,vertex_prj=self.sample_prj_feature(self.smplx_deform_res['vertices'],img_feature_2,batch['w2c_cam'])
        vertex_global_feature=vertex_global_feature[:,None,:].expand(-1,vertex_sample_feature.shape[-2],-1)

        vertex_sample_feature=torch.cat([vertex_sample_feature,vertex_base_feature,vertex_global_feature],dim=-1)
        vertex_gs_dict=self.vertex_gs_decoder(vertex_sample_feature,cam_dirs)
        vertex_gs_dict["positions"]=self.v_template.clone()[None].expand(batch_size,-1,-1)
        
        #uvmap gs
        image_rgb =nn.functional.interpolate(batch['image'],(self.cfg.image_size,self.cfg.image_size),mode='bilinear',align_corners=False)
        img_feature=torch.cat([image_rgb,img_feature],dim=1)
        
        with torch.no_grad():
            visble_faces,fragments=self.mesh_renderer.render_fragments(self.smplx_deform_res['vertices'],transform_matrix=batch['w2c_cam'])
        uvmap_features=self.convert_pixel_feature_to_uv(img_feature,self.smplx_deform_res['vertices'],batch['w2c_cam'],
                                                        visble_faces=visble_faces)

        extra_style=self.uv_style_mapping(global_faeature)
        uvmap_features=self.uv_feature_decoder(uvmap_features,extra_style=extra_style)
        uvmap_features=torch.cat([uvmap_features,self.uv_base_feature[None].expand(batch_size,-1,-1,-1)],dim=1)
        uv_point_gs_dict=self.uv_point_decoder(uvmap_features,cam_dirs)

        for key in uv_point_gs_dict.keys():
            gs_f0=uv_point_gs_dict[key].reshape(batch_size,self.uvmap_size*self.uvmap_size,-1)
            uv_point_gs_dict[key] =gs_f0[:,self.uv_mask_flat,:]
        binding_face=self.smplx.uvmap_f_idx.clone().reshape(1,self.uvmap_size*self.uvmap_size,-1)
        uv_point_gs_dict['binding_face']=binding_face[:,self.uv_mask_flat,:]
        face_bary=self.smplx.uvmap_f_bary.clone().reshape(1,self.uvmap_size*self.uvmap_size,-1)
        uv_point_gs_dict['face_bary']=face_bary[:,self.uv_mask_flat,:]
        extra_dict['uvmap_texture']=torch.sigmoid(uvmap_features[:,:3].permute(0,2,3,1).contiguous())

        return vertex_gs_dict,uv_point_gs_dict,extra_dict

#deform ubody gaussians from canonical space to deformed space
class Ubody_Gaussian(L.LightningModule):
    def __init__(self,cfg,vertex_gaussian_assets,uv_gaussian_assets,pruning=False):
        super().__init__()
        self.cfg=cfg
        self.max_sh_degree=cfg.sh_degree
        self.opacity_threshold=cfg.opacity_threshold
        
        self._smplx_scaling=vertex_gaussian_assets['scales']
        self._smplx_rotation=vertex_gaussian_assets['rotations']
        self._smplx_opacity=vertex_gaussian_assets['opacities']
        self._smplx_xyz=vertex_gaussian_assets['positions']
        self._smplx_offset=vertex_gaussian_assets['static_offsets']

        self._uv_scaling=uv_gaussian_assets['scales']
        self._uv_rotation=uv_gaussian_assets['rotations']
        self._uv_opacity=uv_gaussian_assets['opacities']
        self._uv_local_xyz=uv_gaussian_assets['local_pos']
        
        #same for all batch
        self._uv_binding_face=uv_gaussian_assets['binding_face'][0].squeeze(-1) 
        self._uv_face_bary=uv_gaussian_assets['face_bary'][0].squeeze(-1) 

        self._smplx_features_color=vertex_gaussian_assets['colors']
        self._uv_features_color=uv_gaussian_assets['colors']
        self._smplx_features_color[...,:3]=torch.sigmoid(self._smplx_features_color[...,:3])
        self._uv_features_color[...,:3]=torch.sigmoid(self._uv_features_color[...,:3])
        
        self.ehm = EHM(self.cfg.flame_assets_dir, self.cfg.smplx_assets_dir, add_teeth=True, uv_size=self.cfg.uvmap_size)
        self.smplx = self.ehm.smplx

        self._canoical=False
        if pruning:
            self.prune_gaussians()
            
    def to(self, *args, **kwargs):
        # First, call the original .to() method to move registered parameters and buffers
        super().to(*args, **kwargs)

        # Get the target device from the arguments
        device = args[0] if args else kwargs.get('device')

        # Manually move all the unregistered tensor attributes
        self._smplx_scaling = self._smplx_scaling.to(device)
        self._smplx_rotation = self._smplx_rotation.to(device)
        self._smplx_opacity = self._smplx_opacity.to(device)
        self._smplx_xyz = self._smplx_xyz.to(device)
        if self._smplx_offset is not None:
            self._smplx_offset = self._smplx_offset.to(device)
        self._uv_scaling = self._uv_scaling.to(device)
        self._uv_rotation = self._uv_rotation.to(device)
        self._uv_opacity = self._uv_opacity.to(device)
        self._uv_local_xyz = self._uv_local_xyz.to(device)
        self._uv_binding_face = self._uv_binding_face.to(device)
        self._uv_face_bary = self._uv_face_bary.to(device)
        self._smplx_features_color = self._smplx_features_color.to(device)
        self._uv_features_color = self._uv_features_color.to(device)
        
        # Ensure the nested ehm model is also moved
        if self.ehm is not None:
            self.ehm.to(device)
            
        return self

    def init_ehm(self, ehm_state_dict=None):
        if ehm_state_dict is not None:
            self.ehm.load_state_dict(ehm_state_dict)

    def prune_gaussians(self):
        #prune gaussians with opacity less than threshold
        #assert batch size is 1
        assert self._uv_opacity.shape[0]==1 
        mask=self._uv_opacity>self.opacity_threshold
        mask=mask.squeeze(-1)
        mask_bool = mask[0].bool()
        
        self._uv_scaling=(self._uv_scaling[:,mask_bool])
        self._uv_rotation=(self._uv_rotation[:,mask_bool])
        self._uv_opacity=(self._uv_opacity[:,mask_bool])
        self._uv_local_xyz=(self._uv_local_xyz[:,mask_bool])
        self._uv_binding_face=self._uv_binding_face[mask_bool]
        self._uv_face_bary=self._uv_face_bary[mask_bool]
        self._uv_features_color=(self._uv_features_color[:,mask_bool])
    
    def forward(self,batch):
        #batch: traget pose expr
        #smplx vertex gaussians
        batch_size=self._smplx_xyz.shape[0]
        deformed_assets={}
        smplx_deform_res=self.ehm(batch['smplx_coeffs'],batch['flame_coeffs'])

        self._smplx_xyz_deform=smplx_deform_res["vertices"]
        d_deform_rot_xyzw=rotmat_to_unitquat(smplx_deform_res["ver_transform_mat"][:,:,:3,:3])
        self._smplx_rotation_deform=torch.nn.functional.normalize(quat_xyzw_to_wxyz(quat_product(d_deform_rot_xyzw,quat_wxyz_to_xyzw(self._smplx_rotation))),dim=-1)

        #uvmap gaussians
        face_orien_mat, face_scaling = compute_face_orientation(smplx_deform_res["vertices"], self.smplx.faces_tensor, return_scale=True)
        face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat))
        
        face_vertices=smplx_deform_res["vertices"][:,self.smplx.faces_tensor]# b f k 3
        face_vertices_nn=face_vertices[:,self._uv_binding_face]# b n k 3
        face_bary=self._uv_face_bary[None].expand(batch_size,-1,-1)# b n k
        face_center_nn= torch.einsum('bnk,bnkj->bnj',face_bary,face_vertices_nn)# B n 3
        face_scaling_nn=face_scaling[:,self._uv_binding_face]
        
        xyz=torch.einsum('bnij,bnj->bni',face_orien_mat[:,self._uv_binding_face], self._uv_local_xyz)
        self._uv_xyz_deform=xyz * face_scaling_nn + face_center_nn
        
        face_orien_quat=face_orien_quat[:,self._uv_binding_face]
        self._uv_rotation_deform=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(self._uv_rotation)))
        self._uv_scaling_deform = self._uv_scaling* face_scaling_nn
        
        #integrate
        self._xyz_deform=torch.cat([self._smplx_xyz_deform,self._uv_xyz_deform],dim=1)
        self._rotation_deform=torch.cat([self._smplx_rotation_deform,self._uv_rotation_deform],dim=1)
        self._scaling_deform=torch.cat([self._smplx_scaling,self._uv_scaling_deform],dim=1)
        self._opacity_deform=torch.cat([self._smplx_opacity,self._uv_opacity],dim=1)
        self._features_color=torch.cat([self._smplx_features_color,self._uv_features_color],dim=1)
        deformed_assets.update({'features_color':self._features_color})

        deformed_assets .update( {
        'xyz': self._xyz_deform,
        'rotation': self._rotation_deform, 
        'scaling': self._scaling_deform, 
        'opacity': self._opacity_deform, 
        'sh_degree':self.max_sh_degree,
        'smplx_xyz_deform':smplx_deform_res["vertices"],
        })
        return deformed_assets

    def get_canoical_gaussians(self):
        #uvmap gaussians
        batch_size=self._smplx_xyz.shape[0]
        v_template=self._smplx_xyz.clone().expand(batch_size,-1,-1)
        if self._smplx_offset is not None:
            v_template=v_template+self._smplx_offset
        face_orien_mat, face_scaling = compute_face_orientation(v_template, self.smplx.faces_tensor, return_scale=True)
        face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat))
        
        face_vertices=v_template[:,self.smplx.faces_tensor]# b f k 3
        face_vertices_nn=face_vertices[:,self._uv_binding_face]# b n k 3
        face_bary=self._uv_face_bary[None].expand(batch_size,-1,-1)# b n k
        face_center_nn= torch.einsum('bnk,bnkj->bnj',face_bary,face_vertices_nn)# B n 3
        face_scaling_nn=face_scaling[:,self._uv_binding_face]
        
        xyz=torch.einsum('bnij,bnj->bni',face_orien_mat[:,self._uv_binding_face], self._uv_local_xyz)
        self._uv_xyz_cano=xyz * face_scaling_nn + face_center_nn
        
        face_orien_quat=torch.nn.functional.normalize(face_orien_quat[:,self._uv_binding_face],dim=-1)
        self._uv_rotation_cano=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(self._uv_rotation)))
        self._uv_scaling_cano = self._uv_scaling* face_scaling_nn
        self._uv_opacity_cano=self._uv_opacity
        self._canoical=True

    def save_point_ply(self,save_path,save_split=False,assets=None):
        
        if not self._canoical:
            self.get_canoical_gaussians()
        
        if assets is not None:
            xyz_all_np=assets['xyz'][0].detach().cpu().numpy()
            colors_all_np=assets['features_color'][0,...,:3].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_all_np)
            pcd.colors = o3d.utility.Vector3dVector(colors_all_np)
            o3d.io.write_point_cloud(os.path.join(save_path,'deformed.ply'), pcd)
        else:
            xyz_all_np=torch.cat([self._smplx_xyz,self._uv_xyz_cano],dim=1).detach().cpu().numpy()
            colors_all_np=torch.cat([self._smplx_features_color[...,:3],self._uv_features_color[...,:3]],dim=1).detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_all_np[0])
            pcd.colors = o3d.utility.Vector3dVector(colors_all_np[0])
            o3d.io.write_point_cloud(os.path.join(save_path,'canonical.ply'), pcd)
            
            if save_split:
                xyz_smplx_np=self._smplx_xyz.detach().cpu().numpy()
                colors_smplx_np=self._smplx_features_color[...,:3].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_smplx_np[0])
                pcd.colors = o3d.utility.Vector3dVector(colors_smplx_np[0])
                o3d.io.write_point_cloud(os.path.join(save_path,'canonical_smplx.ply'), pcd)

                xyz_uv_np=self._uv_xyz_cano.detach().cpu().numpy()
                colors_uv_np=self._uv_features_color[...,:3].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_uv_np[0])
                pcd.colors = o3d.utility.Vector3dVector(colors_uv_np[0])
                o3d.io.write_point_cloud(os.path.join(save_path,'canonical_uv.ply'), pcd)
    
    def save_gaussian_ply(self,save_path,save_split=False):
        if not self._canoical:
            self.get_canoical_gaussians()
            
        xyz_all_np=torch.cat([self._smplx_xyz,self._uv_xyz_cano],dim=1)[0].detach().cpu().numpy()
        colors_all_np=torch.cat([self._smplx_features_color[0,:,:3],self._uv_features_color[0,:,:3]],dim=0)[:,None]
        colors_all_np=(colors_all_np) / 0.28209479177387814 #RGB to SHS
        opacities_all_np = torch.cat([inverse_sigmoid(self._smplx_opacity),
                                      inverse_sigmoid(self._uv_opacity_cano)],dim=1)[0].detach().cpu().numpy()
        scale_all_np = torch.cat([torch.log(self._smplx_scaling),
                                  torch.log(self._uv_scaling_cano)],dim=1)[0].detach().cpu().numpy()
        rotation_all_np = torch.cat([self._smplx_rotation,self._uv_rotation_cano],dim=1)[0].detach().cpu().numpy()
        
        normals = np.zeros_like(xyz_all_np)
        f_dc = colors_all_np.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest=torch.zeros((f_dc.shape[0],0,3))
        f_rest=f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz_all_np.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz_all_np, normals, f_dc, f_rest, opacities_all_np, scale_all_np, rotation_all_np), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(save_path, 'GS_canonical.ply'))
        
        if save_split:
            xyz_smplx_np=self._smplx_xyz[0].detach().cpu().numpy()
            colors_smplx_np=self._smplx_features_color[0,...,:3][:,None]
            colors_smplx_np=(colors_smplx_np ) / 0.28209479177387814 #RGB to SHS
            opacities_smplx_np = inverse_sigmoid(self._smplx_opacity[0]).detach().cpu().numpy()
            scale_smplx_np = torch.log(self._smplx_scaling[0]).detach().cpu().numpy()
            rotation_smplx_np = self._smplx_rotation[0].detach().cpu().numpy()
            normals = np.zeros_like(xyz_smplx_np)
            f_dc = colors_smplx_np.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest=torch.empty((f_dc.shape[0],0,3))
            f_rest=f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attributes = np.concatenate((xyz_smplx_np, normals, f_dc, f_rest, opacities_smplx_np, scale_smplx_np, rotation_smplx_np), axis=1)
            elements = np.empty(xyz_smplx_np.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(os.path.join(save_path, 'GS_canonical_smplx.ply'))

            xyz_uv_np=self._uv_xyz_cano[0].detach().cpu().numpy()
            colors_uv_np=self._uv_features_color[0,...,:3][:,None]
            colors_uv_np=(colors_uv_np ) / 0.28209479177387814 #RGB to SHS
            opacities_uv_np = inverse_sigmoid(self._uv_opacity_cano[0]).detach().cpu().numpy()
            scale_uv_np = torch.log(self._uv_scaling_cano[0]).detach().cpu().numpy()
            rotation_uv_np = self._uv_rotation_cano[0].detach().cpu().numpy()
            normals = np.zeros_like(xyz_uv_np)
            f_dc = colors_uv_np.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest=torch.empty((f_dc.shape[0],0,3))
            f_rest=f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attributes = np.concatenate((xyz_uv_np, normals, f_dc, f_rest, opacities_uv_np, scale_uv_np, rotation_uv_np), axis=1)
            elements = np.empty(xyz_uv_np.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(os.path.join(save_path, 'GS_canonical_uv.ply'))
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(0):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l
    
def configure_optimizers(infer_model,cfg,render_model=None):
    learning_rate = cfg.learning_rate
    print('Learning rate: {}'.format(learning_rate))
    decay_names = []
    normal_params, decay_params0, decay_params1 = [], [], []

    def process_model_parameters(model):
        nonlocal decay_names, normal_params, decay_params0, decay_params1
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'style_mlp' in name or 'final_linear' in name:
                decay_names.append(".".join(name.split('.')[:-2]) if len(name.split('.')) > 3 else ".".join(name.split('.')[:-1]))
                decay_params0.append(param)
            elif 'up_point_decoder' in name or ('vertex_gs_decoder' in name and 'feature_layers' not in name) \
                  or 'uv_feature_decoder' in name or 'prj_feature_decoder' in name:
                decay_params1.append(param)
            else:
                normal_params.append(param)
    
    process_model_parameters(infer_model)
    if render_model is not None:
        process_model_parameters(render_model)

    # optimizer
    optimizer = torch.optim.Adam([
            {'params': normal_params, 'lr': learning_rate},
            {'params': decay_params0, 'lr': learning_rate*0.1},
            {'params': decay_params1, 'lr': learning_rate},
        ], lr=learning_rate, betas=(0.0, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=cfg.lr_decay_rate, 
        total_iters=cfg.lr_decay_iter,
    )
    return optimizer, scheduler


def get_cam_dirs(_cam):
    # get the z axis of cam in wold coordinate
    batch_size=_cam.shape[0]
    z_dirs=torch.tensor([[0,0,1]],dtype=torch.float32,device=_cam.device)
    z_dirs=z_dirs.expand(batch_size,-1)
    z_dirs=torch.einsum('bij,bj->bi', _cam[:, :3, :3], z_dirs)#b 3
    return z_dirs

def get_pixel_coordinates(image_height, image_width):
    x_range = torch.arange(0, image_width, dtype=torch.float32)+0.5
    y_range = torch.arange(0, image_height, dtype=torch.float32)+0.5
    coords = torch.cartesian_prod(x_range, y_range)
    coords = coords[:, [1, 0]]
    coords=coords.reshape(image_height,image_width,-1)
    return coords