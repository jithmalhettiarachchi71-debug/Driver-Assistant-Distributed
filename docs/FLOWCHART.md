# Vehicle Safety Alert System - Flowchart

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          VEHICLE SAFETY ALERT SYSTEM                            │
│                              Main Processing Loop                               │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────┐
                                    │   START     │
                                    └──────┬──────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────┐
                    │           INITIALIZATION                  │
                    │  • Load config.yaml                       │
                    │  • Setup camera (CSI/Webcam/Video)       │
                    │  • Load YOLO model (ONNX)                │
                    │  • Initialize lane detection pipeline    │
                    │  • Initialize alert decision engine      │
                    │  • Setup GPIO LEDs & LiDAR (Pi only)     │
                    │  • Start telemetry logging               │
                    └──────────────────┬───────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  Turn ON System LED     │
                         │     (GPIO 17)           │
                         └────────────┬────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            ▼                            │
         │              ┌──────────────────────────┐               │
         │              │    MAIN PROCESSING LOOP   │◄─────────────┤
         │              └─────────────┬────────────┘               │
         │                            │                            │
         │                            ▼                            │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 1: FRAME ACQUISITION                      │    │
         │  │  • Capture frame from camera                    │    │
         │  │  • Timeout handling                             │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 2: YOLO OBJECT DETECTION                  │    │
         │  │  • Run every N frames (frame_skip: 5)           │    │
         │  │  • Preprocess → ONNX inference → Postprocess    │    │
         │  │  • Detect: Traffic lights, Vehicles,            │    │
         │  │            Pedestrians, Bikers, Stop signs      │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 3: LANE DETECTION                         │    │
         │  │  • Color filtering (white/yellow lanes)         │    │
         │  │  • Canny edge detection                         │    │
         │  │  • Hough line transform                         │    │
         │  │  • Polynomial curve fitting                     │    │
         │  │  • Temporal smoothing (EMA)                     │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 4: DANGER ZONE EVALUATION                 │    │
         │  │  • Update trapezoidal zone from lanes           │    │
         │  │  • Check if detections are in danger zone       │    │
         │  │  • Identify collision risk candidates           │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 5: LiDAR DISTANCE CHECK                   │    │
         │  │  • Read TF-Luna distance (UART)                 │    │
         │  │  • Update decision engine with distance         │    │
         │  │  • Collision requires distance < 600cm          │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 6: ALERT DECISION                         │    │
         │  │  • Evaluate all hazards                         │    │
         │  │  • Apply priority system                        │    │
         │  │  • Check cooldowns                              │    │
         │  │  • Generate AlertEvent if conditions met        │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │           ┌─────────────┴─────────────┐                 │
         │           ▼                           ▼                 │
         │    ┌─────────────┐             ┌─────────────┐          │
         │    │ Alert = Yes │             │ Alert = No  │          │
         │    └──────┬──────┘             └──────┬──────┘          │
         │           │                           │                 │
         │           ▼                           │                 │
         │  ┌────────────────────┐               │                 │
         │  │ STAGE 7: DISPATCH  │               │                 │
         │  │  • Play audio beep │               │                 │
         │  │  • Trigger buzzer  │               │                 │
         │  │  • Alert LED ON    │               │                 │
         │  └─────────┬──────────┘               │                 │
         │            │                          │                 │
         │            └──────────┬───────────────┘                 │
         │                       │                                 │
         │                       ▼                                 │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 8: GPIO OUTPUT                            │    │
         │  │  • Set Alert LED (GPIO 27)                      │    │
         │  │  • Set Collision Output (GPIO 22) if in zone    │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 9: DISPLAY RENDERING (Optional)           │    │
         │  │  • Draw detection bounding boxes                │    │
         │  │  • Draw lane line overlays                      │    │
         │  │  • Draw danger zone trapezoid                   │    │
         │  │  • Draw alert banner (with persistence)         │    │
         │  │  • Draw info panel (FPS, stats)                 │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │  ┌─────────────────────────────────────────────────┐    │
         │  │ STAGE 10: TELEMETRY                             │    │
         │  │  • Log frame metrics to JSONL                   │    │
         │  │  • Record latencies, detections, alerts         │    │
         │  └──────────────────────┬──────────────────────────┘    │
         │                         │                               │
         │                         ▼                               │
         │               ┌──────────────────┐                      │
         │               │  Quit Signal?    │                      │
         │               └────────┬─────────┘                      │
         │                        │                                │
         │           ┌────────────┴────────────┐                   │
         │           ▼                         ▼                   │
         │        ┌─────┐                   ┌─────┐                │
         │        │ No  │───────────────────│ Yes │                │
         │        └─────┘                   └──┬──┘                │
         │           │                         │                   │
         └───────────┘                         ▼
                                    ┌──────────────────┐
                                    │     CLEANUP      │
                                    │  • Stop LiDAR    │
                                    │  • Turn off LEDs │
                                    │  • Release GPIO  │
                                    │  • Close camera  │
                                    │  • Stop logging  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                        ┌─────────┐
                                        │   END   │
                                        └─────────┘
