from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

process_pool = {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT)
env["PYTHONUNBUFFERED"] = "1"

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


async def run_stage(websocket: WebSocket, stage: int, prompt: str):
    target = prompt.replace(" ", "_")
    output_path = PROJECT_ROOT / "output"
    output_folder = "example_generation/" + target

    stage_scripts = {
        1: [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "1_template_generation/main.py"),
            "--target", target,
            "--output_path", str(output_path),
            "--output_folder", output_folder,
            "--system_path", str(PROJECT_ROOT),
            "--model", "claude-3-5-sonnet-20240620",
            "--prompt", prompt
        ],
        2: [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "2_detail_enhancement/main.py"),
            "--target", target,
            "--output_path", str(output_path),
            "--output_folder", output_folder,
            "--seed", "0",
            "--num_images_per_prompt", "4",
            "--strength", "1.0",
            "--thresh_iou", "0.4",
            "--prompt", prompt,
            "--diffusion_model_id", str(PROJECT_ROOT / "2_detail_enhancement/models/aamXLAnimeMix_v10.safetensors"),
            "--sam_checkpoint", str(PROJECT_ROOT / "2_detail_enhancement/models/sam_vit_h_4b8939.pth")
        ],
        3: [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "3_svg_optimization/main.py"),
            "--target", target,
            "--svg_folder", str(output_path) + "/" + str(output_folder),
            "--system_path", str(PROJECT_ROOT / "3_svg_optimization"),
            "--output_size", "256",
            "--smoothness_weight_img", "2.0",
            "--mse_loss_weight_img", "2000.0",
            "--kl_weight_img", "0.2",
            "--vae_optim_config", str(PROJECT_ROOT / "3_svg_optimization/configs/vae_config_cmd_10.yaml"),
            "--vae_pretrained_path", str(PROJECT_ROOT / "3_svg_optimization/vae_model/cmd_10.pth")
        ]
    }

    try:
        proc = await asyncio.create_subprocess_exec(
            *stage_scripts[stage],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT)
        )

        process_pool[stage] = proc

        while True:
            try:
                line = await proc.stdout.read(1024)
                if not line:
                    if proc.returncode is not None:
                        break
                    await asyncio.sleep(0.1)
                    continue

                decoded = line.decode('utf-8', errors='replace')
                decoded = decoded.replace('\r\n', '\n').rstrip('\n')

                for output_line in decoded.split('\n'):
                    output_line = output_line.strip()
                    if not output_line:
                        continue

                    await websocket.send_json({
                        "stage": stage,
                        "status": "running",
                        "output": output_line
                    })

                    if stage == 1 and "The best SVG is:" in output_line:
                        svg_filename = output_line.split(": ")[-1].strip()
                        await websocket.send_json({
                            "stage": 1,
                            "status": "svg_ready",
                            "svg_file": output_folder + "/stage_1/svg_logs/" + svg_filename
                        })

                    if stage == 1 and output_line == "Stage 1 Done!":
                        await websocket.send_json({
                            "stage": 1,
                            "status": "completed",
                            "output": "Stage 1 Finished"
                        })

                    if stage == 2 and "The best SVG is:" in output_line:
                        svg_filename = output_line.split(": ")[-1].strip()
                        await websocket.send_json({
                            "stage": 2,
                            "status": "svg_ready2",
                            "svg_file": svg_filename
                        })

                    if stage == 2 and output_line == "Stage 2 Done!":
                        await websocket.send_json({
                            "stage": 2,
                            "status": "completed",
                            "output": "Stage 2 Finished"
                        })

                    if stage == 3 and "The best SVG is:" in output_line:
                        svg_filename = output_line.split(": ")[-1].strip()
                        await websocket.send_json({
                            "stage": 3,
                            "status": "svg_ready3",
                            "svg_file": svg_filename
                        })

                    if stage == 3 and output_line == "Stage 3 Done!":
                        await websocket.send_json({
                            "stage": 3,
                            "status": "completed",
                            "output": "Stage 3 Finished"
                        })

            except (asyncio.CancelledError, ConnectionResetError):
                proc.terminate()
                raise
            except Exception as e:
                print(f"error: {str(e)}")
                break

        exit_code = await proc.wait()

        if exit_code != 0:
            await websocket.send_json({
                "stage": stage,
                "status": "error",
                "output": f"Stage {stage} failed with code {exit_code}"
            })
        else:
            await websocket.send_json({
                "stage": stage,
                "status": "completed"
            })

    except asyncio.CancelledError:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
        raise
    except Exception as e:
        await websocket.send_json({
            "stage": stage,
            "status": "error",
            "output": f"Stage {stage} crashed: {str(e)}"
        })


@app.websocket("/ws/generate")
async def websocket_generator(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        params = json.loads(data)
        prompt = params["prompt"]
        start_from = params.get("startFrom", 1)
        stages_to_run = [s for s in [1, 2, 3] if s >= start_from]

        for stage in stages_to_run:
            await websocket.send_json({
                "stage": stage,
                "status": "started",
                "output": f"Stage {stage} started"
            })

            try:
                await asyncio.wait_for(
                    run_stage(websocket, stage, prompt),
                    timeout=2400
                )
            except asyncio.TimeoutError:
                proc = process_pool.get(stage)
                if proc and proc.returncode is None:
                    proc.terminate()
                    await proc.wait()

                await websocket.send_json({
                    "stage": stage,
                    "status": "error",
                    "output": f"Stage {stage} timed out after 40 minutes"
                })
                break

    except json.JSONDecodeError:
        await websocket.send_json({
            "error": "Invalid JSON format"
        })
    except Exception as e:
        await websocket.send_json({
            "error": str(e)
        })
    finally:
        await websocket.close()


@app.get("/api/svg/{file_path:path}")
async def get_svg(file_path: str):
    svg_path = PROJECT_ROOT / "output" / file_path

    # Security checks
    if any(c in file_path for c in {'..', '//', '\\', ':'}):
        raise HTTPException(400, "Illegal file path")

    if not svg_path.exists():
        raise HTTPException(404, "File not found")

    if svg_path.suffix.lower() != '.svg':
        raise HTTPException(400, "Not an SVG file")

    return {
        "status": "ok",
        "content": svg_path.read_text(encoding='utf-8')
    }
