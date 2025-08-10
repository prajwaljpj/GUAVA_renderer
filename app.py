import os
import time
import gradio as gr
from pathlib import Path
from functools import partial
import subprocess

OUTPUT_DIR = 'outputs/app'
OUTNAME = 'render'
DEVICES = '0'

TRACKED_IMG_DIR = 'assets/example/tracked_image'
TRACKED_VID_DIR = 'assets/example/tracked_video'
tracked_images_list = os.listdir(TRACKED_IMG_DIR)
tracked_videos_list = os.listdir(TRACKED_VID_DIR)

def run_cmd(command, current_dir=None):
    print(f"▶️ Executing command:\n{command}", flush=True)
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=current_dir,
        bufsize=1 
    )
    
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        process.stdout.close()

    return_code = process.wait()

    if return_code != 0:
        print(f"‼️ Command failed with return code {return_code}", flush=True)
        if process.stderr:
            error_output = process.stderr.read()
            print(f"Error output:\n{error_output}", flush=True)
            process.stderr.close()
        raise subprocess.CalledProcessError(return_code, command)
        
    print(f"✅ Command executed successfully.", flush=True)


def check_process_status(source_image, driven_video):
    """Check if the processing is complete and return the result video if available"""
    if not OUTPUT_DIR or not os.path.exists(OUTPUT_DIR):
        return "Processing hasn't started yet.", None
    
    src_name = os.path.splitext(os.path.basename(source_image))[0]
    dst_name = os.path.splitext(os.path.basename(driven_video))[0]
        
    output_file =  os.path.join(OUTPUT_DIR, f'{OUTNAME}_cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
    print('Try to find => ' + output_file)

    if not os.path.exists(output_file):
        return "Still processing... You can leave but keep this page open. ⏳", None
        
    return "Processing completed successfully! 🎉", output_file

def generate_from_selection(source_selection, driven_selection, progress=gr.Progress()):

    if not source_selection or not driven_selection:
        return "Please select a source character and a driving video.", None, None
    
    progress(0.01, desc="Detection of selection, preparing to start generation....")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, OUTPUT_DIR)
        my_run_cmd = partial(run_cmd, current_dir=current_dir)
        
        src_img_root = os.path.join(current_dir, TRACKED_IMG_DIR, source_selection['caption'])
        dcv_vid_root = os.path.join(current_dir, TRACKED_VID_DIR, driven_selection['caption'])
        
        if not os.path.exists(src_img_root):
            return f"Error: Unable to find source character data path {src_img_root}", None
        if not os.path.exists(dcv_vid_root):
            return f"Error: Unable to find driven video data path {dcv_vid_root}", None

        print('⚡️ Initiating GUAVA generation results, please wait...')
        src_name = source_selection['caption']
        dst_name = driven_selection['caption']
        output_file = os.path.join(output_dir, f'{OUTNAME}_cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
        
        if os.path.exists(output_file):
            print(f'🐶 The result already exists, skipping generation....')
        else:
            my_run_cmd(f'PYTHONPATH=.  python main/test.py -d {DEVICES} -n {OUTNAME} -m assets/GUAVA' + 
                       f' --source_data_path {src_img_root}'+
                       f' --data_path {dcv_vid_root} ' +
                       f' --save_path {output_dir} ' +
                       f' --skip_self_act --render_cross_act')
            print(f'Completion! The results are saved in {output_dir}/{OUTNAME}_cross_act')
        progress(1.0, desc="🎉 complete! ")
        
        return "🎉 complete! ", output_file
            
    except Exception as e:
        return f"An error occurred: {str(e)}", None
    
def get_history_videos():
    """Get all previously generated videos from results directory"""
    results = []
    base_dir = F"{OUTPUT_DIR}/{OUTNAME}_cross_act"
    if not os.path.exists(base_dir):
        return "No history found", None
        
    for source_dir in os.listdir(base_dir):
        source_path = os.path.join(base_dir, source_dir)
        if os.path.isdir(source_path):
            for driven_dir in os.listdir(source_path):
                driven_path = os.path.join(source_path, driven_dir)
                if os.path.isdir(driven_path):
                    videos = list(Path(driven_path).glob("*.mp4"))
                    for video in videos:
                        results.append({
                            "source": source_dir,
                            "driven": driven_dir,
                            "video": str(video)
                        })
    
    if not results:
        return "No history found", None
    
    # Format the results for gallery display
    video_paths = []
    video_labels = []
    for result in results:
        video_paths.append(result["video"])
        video_labels.append(f"Source: {result['source']} | Driven: {result['driven'][len(result['source'])+1:]}")
        
    return f"Found {len(video_paths)} historical results", (video_paths, video_labels)

def update_history():
    message, gallery_data = get_history_videos()
    if gallery_data is None:
        return message, ""
    
    video_paths, video_labels = gallery_data
    html_content = "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; padding: 20px;'>"
    
    for video_path, video_label in zip(video_paths, video_labels):
        html_content += f"""
        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px;'>
            <video width='100%' controls>
                <source src='file/{video_path}' type='video/mp4'>
                Your browser does not support the video tag.
            </video>
            <div style='margin-top: 10px; font-weight: bold;'>{video_label}</div>
        </div>
        """
    
    html_content += "</div>"
    return message, html_content

def prepare_gallery_data(base_dir):
    gallery_list = []
    if not os.path.exists(base_dir):
        return gallery_list
        
    for item_name in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item_name)
        if os.path.isdir(item_path):
            preview_path = os.path.join(item_path, 'preview.png')
            if os.path.exists(preview_path):
                gallery_list.append((preview_path, item_name))
    return gallery_list
    
'''  
def process_avatar(source_image, driven_video, progress=gr.Progress()):
    if source_image is None or driven_video is None:
        return "Please upload both source image and driven video.", None, None
    
    progress(0.01, desc="Files saved, starting processing...")
    
    try:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, OUTPUT_DIR)

        my_run_cmd = partial(run_cmd, current_dir=current_dir)

        processed_source_image_dir = os.path.join(output_dir, 'processed_source_image')
        processed_driven_video_dir = os.path.join(output_dir, 'processed_driven_video')

        driven_video_fp = driven_video
        source_image_fp = source_image

        os.makedirs(processed_source_image_dir, exist_ok=True)
        os.makedirs(processed_driven_video_dir, exist_ok=True)
        src_img_root = os.path.join(processed_source_image_dir, os.path.basename(source_image_fp).split('.')[0])
        dcv_vid_root = os.path.join(processed_driven_video_dir, os.path.basename(driven_video_fp).split('.')[0])

        # process source image and driven video
        os.chdir(f'{current_dir}/tools/ehm-tracker')
        print('☕️ Processing source image, please wait...')
        if os.path.exists(os.path.join(src_img_root, 'optim_tracking_ehm.pkl')):
            print(f'🐶 Source image has been processed, skipping...')
        else:
            my_run_cmd(f'python prepare_data.py -i {source_image_fp} -o {processed_source_image_dir} --no-save_vis')
        
        progress(0.3, desc="🐶 Source image has been processed")
        
        print('☕️ Processing driven video, please wait...')
        if os.path.exists(os.path.join(dcv_vid_root, 'optim_tracking_ehm.pkl')):
            print(f'🐶 Driven video has been processed, skipping...')
        else:
            my_run_cmd(f'python prepare_data.py -i {driven_video_fp} -o {processed_driven_video_dir} --no-save_vis')
        os.chdir(current_dir)

        progress(0.8, desc="🐶 Driven video has been processed")

        # now driven the result
        print('⚡️ Now driving the result, please wait...')
        src_name = os.path.splitext(os.path.basename(source_image))[0]
        dst_name = os.path.splitext(os.path.basename(driven_video))[0]
        output_file =  os.path.join(output_dir, 'cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
        if os.path.exists(output_file):
            print(f'🐶 Result has been generated, skipping...')
        else:
            my_run_cmd(f'PYTHONPATH=.  python main/test.py -s {src_img_root} ' + 
                                            f' --data_path {dcv_vid_root} ' +
                                            f' --output_dir {output_dir} -d {DEVICES}' +
                                            f' --skip_self_act --render_cross_act')
            print(f'Done! The result is saved in {output_dir}/cross_act')

        progress(1.0, desc="🎉 Done! ")
        output_file =  os.path.join(output_dir, 'cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
        
        return "🎉 Done! ", output_file
                
    except Exception as e:
        return f"Error occurred: {str(e)}", None
''' 

'''
# Create the Gradio interface
with gr.Blocks(title="GUAVA Avatar Generator", css="""
    .image-container { position: relative; display: inline-block; }
    .overlay-button { 
        position: absolute !important; 
        top: 50% !important; 
        left: 50% !important; 
        transform: translate(-50%, -50%) !important;
        opacity: 0;
        transition: opacity 0.3s;
        background: rgba(0,0,0,0.7) !important;
        color: white !important;
        border: none !important;
    }
    .image-container:hover .overlay-button { 
        opacity: 1; 
    }
    .scrollable-column {
        height: 600px !important;
        overflow-y: auto !important;
        padding-right: 10px;
    }
    .scrollable-column::-webkit-scrollbar {
        width: 8px;
    }
    .scrollable-column::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .scrollable-column::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .scrollable-column::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
""") as demo:
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("figures/guava.png", show_label=False, height=100, container=False, interactive=False)
        with gr.Column(scale=5):
            gr.Markdown("""
            # GUAVA: Generalizable Upper Body 3D Gaussian Avatar
            Upload a source image and a driving video to generate an animated 3D Upperbody avatar.
            """)
    
    with gr.Row():
        with gr.Column():
            source_image = gr.Image(label="Source Image", type="filepath", height=600)
            driven_video = gr.Video(label="Driving Video")
            process_btn = gr.Button("Animate Avatar", variant="primary")
        
        with gr.Column():
            output_message = gr.Textbox(label="Status (Processing takes ~a few mins, you can leave but keep this page open)")
            output_video = gr.Video(label="Generated Animation")
            check_btn = gr.Button("Check Progress 🔄", variant="secondary")
            
            with gr.Row():
                process_btn.click(
                        fn=process_avatar,
                        inputs=[source_image, driven_video],
                        outputs=[output_message, output_video]
                    )
                
                check_btn.click(
                    fn=check_process_status,
                    inputs=[source_image, driven_video],
                    outputs=[output_message, output_video]
                )
    
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Example Source Images (Click to use)")
            with gr.Row(elem_classes=["scrollable-column"]):
                example_images = [f for f in os.listdir("assets/demo/images") if f.endswith(('.png', '.jpg', '.jpeg'))]
                for img in example_images:
                    img_path = f"assets/demo/images/{img}"
                    with gr.Column(elem_classes=["image-container"]):
                        gr.Image(value=img_path, show_label=True, label=img, height=250)
                        select_img_btn = gr.Button("Use this image", size="sm", elem_classes=["overlay-button"])
                        select_img_btn.click(
                            fn=lambda x=img_path: x,
                            outputs=[source_image]
                        )

        with gr.Column(scale=1):
            gr.Markdown("### Example Driving Videos")
            example_videos = [f for f in os.listdir("assets/demo/videos") if f.endswith(('.mp4', '.avi', '.mov'))]
            with gr.Row(elem_classes=["scrollable-column"]):
                for video in example_videos:
                    video_path = f"assets/demo/videos/{video}"
                    with gr.Column():
                        gr.Video(value=video_path, show_label=False, width=300)
                        select_vid_btn = gr.Button("Use this video ➡️", size="sm")
                        select_vid_btn.click(
                            fn=lambda x=video_path: x,
                            outputs=[driven_video]
                        )
    
    gr.Markdown("---")
    with gr.Row(elem_classes=["scrollable-column"]):
        with gr.Column():
            gr.Markdown("### History Viewer")
            history_message = gr.Textbox(label="History Status")
            view_history_btn = gr.Button("View History 📜", variant="secondary")
            history_html = gr.HTML()
            
            view_history_btn.click(
                fn=update_history,
                inputs=[],
                outputs=[history_message, history_html]
            )
'''

with gr.Blocks(title="GUAVA Avatar Generator", css="""...""") as demo:
    source_gallery_data = prepare_gallery_data(TRACKED_IMG_DIR)
    video_gallery_data = prepare_gallery_data(TRACKED_VID_DIR)
    selected_source = gr.State(None)
    selected_video = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("assets/Docs/guava.png", show_label=False, height=100, container=False, interactive=False)
        with gr.Column(scale=5):
            gr.Markdown("""
            # GUAVA: Generalizable Upper Body 3D Gaussian Avatar
            **[Demo Version]** Please select a character and a driving video from the galleries below to generate an animated avatar.
            """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Select a Source Character")
            source_gallery = gr.Gallery(
                value=source_gallery_data, 
                label="Source Characters", 
                columns=3, 
                height="auto",
                object_fit="contain"
            )
        with gr.Column():
            gr.Markdown("### 2. Select a Driving Video")
            video_gallery = gr.Gallery(
                value=video_gallery_data, 
                label="Driving Videos", 
                columns=3, 
                height="auto",
                object_fit="contain"
            )
            
    with gr.Row():
        with gr.Column(scale=1):
            process_btn = gr.Button("Generate Animated Avatar", variant="primary")
            check_btn = gr.Button("Check Progress 🔄", variant="secondary")
        with gr.Column(scale=2):
            output_message = gr.Textbox(label="Status (Processing takes ~a few mins. You can leave but keep this page open)")
            output_video = gr.Video(label="Generated Animation", width=500)
            
    def handle_source_select(evt: gr.SelectData):
        print(f"Selected source character: {evt.value}")
        return evt.value

    def handle_video_select(evt: gr.SelectData):
        print(f"Selected driving video: {evt.value}")
        return evt.value

    source_gallery.select(
        fn=handle_source_select,
        inputs=None,
        outputs=selected_source
    )
    
    video_gallery.select(
        fn=handle_video_select,
        inputs=None,
        outputs=selected_video
    )

    process_btn.click(
        fn=generate_from_selection,
        inputs=[selected_source, selected_video],
        outputs=[output_message, output_video]
    )
    
    check_btn.click(
        fn=check_process_status,
        inputs=[selected_source, selected_video],
        outputs=[output_message, output_video]
    )

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📜 History Viewer")
            history_message = gr.Textbox(label="History Status", interactive=False)
            view_history_btn = gr.Button("View Generation History")
            history_html = gr.HTML() 
            
            view_history_btn.click(
                fn=update_history,
                inputs=[],
                outputs=[history_message, history_html]
            )
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Whether to share the app.")
    args = parser.parse_args()

    demo.launch(server_name="0.0.0.0", allowed_paths=[".",OUTPUT_DIR], share=args.share,)