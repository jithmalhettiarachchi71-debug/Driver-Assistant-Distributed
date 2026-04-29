#!/usr/bin/env python3
"""
Telemetry Analysis Tool for Driver Assistant

Analyzes telemetry.jsonl files and generates performance metrics
with graphs saved as PNG files.

Usage:
    python tools/analyze_telemetry.py [telemetry_file] [--output-dir DIR]
    
Examples:
    python tools/analyze_telemetry.py telemetry.jsonl
    python tools/analyze_telemetry.py telemetry.jsonl --output-dir reports/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


@dataclass
class TelemetryStats:
    """Statistics computed from telemetry data."""
    total_frames: int = 0
    duration_seconds: float = 0.0
    
    # FPS stats
    fps_mean: float = 0.0
    fps_std: float = 0.0
    fps_min: float = 0.0
    fps_max: float = 0.0
    
    # Latency stats (ms)
    lane_latency_mean: float = 0.0
    lane_latency_std: float = 0.0
    lane_latency_p95: float = 0.0
    lane_latency_max: float = 0.0
    
    yolo_latency_mean: float = 0.0
    yolo_latency_std: float = 0.0
    yolo_latency_p95: float = 0.0
    yolo_latency_max: float = 0.0
    
    capture_latency_mean: float = 0.0
    capture_latency_p95: float = 0.0
    
    # Detection stats
    lane_valid_ratio: float = 0.0
    yolo_skip_ratio: float = 0.0
    avg_detections: float = 0.0
    collision_risk_frames: int = 0
    
    # Alert stats
    alert_counts: Dict[str, int] = field(default_factory=dict)
    
    # LiDAR stats
    lidar_valid_ratio: float = 0.0
    lidar_distance_mean: float = 0.0
    lidar_distance_min: float = 0.0
    
    # IP Camera stats
    ip_latency_mean: float = 0.0
    ip_reconnects: int = 0
    dropped_frames: int = 0


def load_telemetry(filepath: Path) -> pd.DataFrame:
    """Load telemetry JSONL file into a DataFrame."""
    records = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    if not records:
        raise ValueError(f"No valid telemetry records found in {filepath}")
    
    df = pd.DataFrame(records)
    
    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} telemetry records from {filepath}")
    return df


def compute_stats(df: pd.DataFrame) -> TelemetryStats:
    """Compute statistics from telemetry DataFrame."""
    stats = TelemetryStats()
    
    stats.total_frames = len(df)
    
    # Duration
    if 'timestamp' in df.columns and len(df) > 1:
        stats.duration_seconds = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    
    # FPS stats
    if 'capture_fps' in df.columns:
        fps = df['capture_fps'].dropna()
        if len(fps) > 0:
            stats.fps_mean = fps.mean()
            stats.fps_std = fps.std()
            stats.fps_min = fps.min()
            stats.fps_max = fps.max()
    
    # Lane latency
    if 'lane_latency_ms' in df.columns:
        lane_lat = df['lane_latency_ms'].dropna()
        if len(lane_lat) > 0:
            stats.lane_latency_mean = lane_lat.mean()
            stats.lane_latency_std = lane_lat.std()
            stats.lane_latency_p95 = lane_lat.quantile(0.95)
            stats.lane_latency_max = lane_lat.max()
    
    # YOLO latency (excluding skipped frames)
    if 'yolo_latency_ms' in df.columns:
        yolo_lat = df['yolo_latency_ms'].dropna()
        if len(yolo_lat) > 0:
            stats.yolo_latency_mean = yolo_lat.mean()
            stats.yolo_latency_std = yolo_lat.std()
            stats.yolo_latency_p95 = yolo_lat.quantile(0.95)
            stats.yolo_latency_max = yolo_lat.max()
    
    # Capture latency
    if 'capture_latency_ms' in df.columns:
        cap_lat = df['capture_latency_ms'].dropna()
        if len(cap_lat) > 0:
            stats.capture_latency_mean = cap_lat.mean()
            stats.capture_latency_p95 = cap_lat.quantile(0.95)
    
    # Lane valid ratio
    if 'lane_valid' in df.columns:
        stats.lane_valid_ratio = df['lane_valid'].sum() / len(df)
    
    # YOLO skip ratio
    if 'yolo_skipped' in df.columns:
        stats.yolo_skip_ratio = df['yolo_skipped'].sum() / len(df)
    
    # Average detections
    if 'detections_count' in df.columns:
        stats.avg_detections = df['detections_count'].mean()
    
    # Collision risk frames
    if 'collision_risks' in df.columns:
        stats.collision_risk_frames = (df['collision_risks'] > 0).sum()
    
    # Alert counts
    if 'alert_type' in df.columns:
        alert_counts = df['alert_type'].dropna().value_counts().to_dict()
        stats.alert_counts = {str(k): int(v) for k, v in alert_counts.items()}
    
    # LiDAR stats
    if 'lidar_valid' in df.columns:
        stats.lidar_valid_ratio = df['lidar_valid'].sum() / len(df)
    
    if 'lidar_distance_cm' in df.columns:
        lidar_dist = df['lidar_distance_cm'].dropna()
        if len(lidar_dist) > 0:
            stats.lidar_distance_mean = lidar_dist.mean()
            stats.lidar_distance_min = lidar_dist.min()
    
    # IP Camera stats
    if 'ip_acquisition_latency_ms' in df.columns:
        ip_lat = df['ip_acquisition_latency_ms'].dropna()
        if len(ip_lat) > 0:
            stats.ip_latency_mean = ip_lat.mean()
    
    if 'ip_reconnect_count' in df.columns:
        stats.ip_reconnects = df['ip_reconnect_count'].max()
    
    if 'dropped_frames' in df.columns:
        stats.dropped_frames = df['dropped_frames'].max()
    
    return stats


def print_stats(stats: TelemetryStats) -> None:
    """Print statistics to console."""
    print("\n" + "=" * 60)
    print("TELEMETRY ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"\n📊 Session Overview:")
    print(f"   Total Frames: {stats.total_frames:,}")
    print(f"   Duration: {stats.duration_seconds:.1f} seconds ({stats.duration_seconds/60:.1f} min)")
    
    print(f"\n⚡ FPS Performance:")
    print(f"   Mean: {stats.fps_mean:.1f} FPS")
    print(f"   Std Dev: {stats.fps_std:.2f}")
    print(f"   Range: {stats.fps_min:.1f} - {stats.fps_max:.1f} FPS")
    
    print(f"\n⏱️ Latency (milliseconds):")
    print(f"   Lane Detection:  mean={stats.lane_latency_mean:.1f}, p95={stats.lane_latency_p95:.1f}, max={stats.lane_latency_max:.1f}")
    print(f"   YOLO Inference:  mean={stats.yolo_latency_mean:.1f}, p95={stats.yolo_latency_p95:.1f}, max={stats.yolo_latency_max:.1f}")
    print(f"   Frame Capture:   mean={stats.capture_latency_mean:.1f}, p95={stats.capture_latency_p95:.1f}")
    if stats.ip_latency_mean > 0:
        print(f"   IP Acquisition:  mean={stats.ip_latency_mean:.1f}")
    
    print(f"\n🛣️ Detection Stats:")
    print(f"   Lane Valid: {stats.lane_valid_ratio*100:.1f}%")
    print(f"   YOLO Skip Rate: {stats.yolo_skip_ratio*100:.1f}%")
    print(f"   Avg Detections/Frame: {stats.avg_detections:.2f}")
    print(f"   Collision Risk Frames: {stats.collision_risk_frames}")
    
    if stats.alert_counts:
        print(f"\n🚨 Alerts:")
        for alert_type, count in sorted(stats.alert_counts.items()):
            print(f"   {alert_type}: {count}")
    
    if stats.lidar_valid_ratio > 0:
        print(f"\n📡 LiDAR:")
        print(f"   Valid Readings: {stats.lidar_valid_ratio*100:.1f}%")
        print(f"   Mean Distance: {stats.lidar_distance_mean:.1f} cm")
        print(f"   Min Distance: {stats.lidar_distance_min:.1f} cm")
    
    if stats.dropped_frames > 0 or stats.ip_reconnects > 0:
        print(f"\n⚠️ Issues:")
        print(f"   Dropped Frames: {stats.dropped_frames}")
        print(f"   IP Reconnects: {stats.ip_reconnects}")
    
    print("\n" + "=" * 60)


def generate_graphs(df: pd.DataFrame, stats: TelemetryStats, output_dir: Path) -> List[Path]:
    """Generate performance graphs and save as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. FPS Over Time
    if 'capture_fps' in df.columns and 'timestamp' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['timestamp'], df['capture_fps'], linewidth=0.8, alpha=0.7)
        ax.axhline(y=stats.fps_mean, color='r', linestyle='--', label=f'Mean: {stats.fps_mean:.1f} FPS')
        ax.set_xlabel('Time')
        ax.set_ylabel('FPS')
        ax.set_title('Frame Rate Over Time')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = output_dir / 'fps_over_time.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        generated_files.append(filepath)
        print(f"  ✓ {filepath.name}")
    
    # 2. Latency Distribution
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    if 'lane_latency_ms' in df.columns:
        lane_lat = df['lane_latency_ms'].dropna()
        axes[0].hist(lane_lat, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=stats.lane_latency_mean, color='r', linestyle='--', label=f'Mean: {stats.lane_latency_mean:.1f}ms')
        axes[0].axvline(x=stats.lane_latency_p95, color='orange', linestyle='--', label=f'P95: {stats.lane_latency_p95:.1f}ms')
        axes[0].set_xlabel('Latency (ms)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Lane Detection Latency')
        axes[0].legend(fontsize=8)
    
    if 'yolo_latency_ms' in df.columns:
        yolo_lat = df['yolo_latency_ms'].dropna()
        if len(yolo_lat) > 0:
            axes[1].hist(yolo_lat, bins=50, edgecolor='black', alpha=0.7, color='green')
            axes[1].axvline(x=stats.yolo_latency_mean, color='r', linestyle='--', label=f'Mean: {stats.yolo_latency_mean:.1f}ms')
            axes[1].axvline(x=stats.yolo_latency_p95, color='orange', linestyle='--', label=f'P95: {stats.yolo_latency_p95:.1f}ms')
            axes[1].set_xlabel('Latency (ms)')
            axes[1].set_title('YOLO Inference Latency')
            axes[1].legend(fontsize=8)
    
    if 'ip_acquisition_latency_ms' in df.columns:
        ip_lat = df['ip_acquisition_latency_ms'].dropna()
        if len(ip_lat) > 0:
            axes[2].hist(ip_lat, bins=50, edgecolor='black', alpha=0.7, color='purple')
            axes[2].axvline(x=stats.ip_latency_mean, color='r', linestyle='--', label=f'Mean: {stats.ip_latency_mean:.1f}ms')
            axes[2].set_xlabel('Latency (ms)')
            axes[2].set_title('IP Camera Latency')
            axes[2].legend(fontsize=8)
    
    plt.tight_layout()
    filepath = output_dir / 'latency_distribution.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    generated_files.append(filepath)
    print(f"  ✓ {filepath.name}")
    
    # 3. Detection Stats Over Time (Rolling Average)
    if 'lane_valid' in df.columns and 'timestamp' in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Rolling lane validity
        window = min(100, len(df) // 10) if len(df) > 10 else 1
        df['lane_valid_rolling'] = df['lane_valid'].rolling(window=window, min_periods=1).mean() * 100
        axes[0].plot(df['timestamp'], df['lane_valid_rolling'], linewidth=1.5, color='green')
        axes[0].axhline(y=stats.lane_valid_ratio * 100, color='r', linestyle='--', 
                        label=f'Overall: {stats.lane_valid_ratio*100:.1f}%')
        axes[0].set_ylabel('Lane Valid (%)')
        axes[0].set_title(f'Lane Detection Reliability (Rolling Window: {window} frames)')
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        
        # Detections count
        if 'detections_count' in df.columns:
            df['detections_rolling'] = df['detections_count'].rolling(window=window, min_periods=1).mean()
            axes[1].plot(df['timestamp'], df['detections_rolling'], linewidth=1.5, color='blue')
            axes[1].axhline(y=stats.avg_detections, color='r', linestyle='--',
                           label=f'Mean: {stats.avg_detections:.2f}')
            axes[1].set_ylabel('Detections Count')
            axes[1].set_title('Object Detections Over Time')
            axes[1].legend()
        
        axes[1].set_xlabel('Time')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = output_dir / 'detection_reliability.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        generated_files.append(filepath)
        print(f"  ✓ {filepath.name}")
    
    # 4. LiDAR Distance Over Time
    if 'lidar_distance_cm' in df.columns and 'timestamp' in df.columns:
        lidar_data = df[df['lidar_valid'] == True] if 'lidar_valid' in df.columns else df
        if len(lidar_data) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(lidar_data['timestamp'], lidar_data['lidar_distance_cm'], 
                   linewidth=0.8, alpha=0.7, color='teal')
            ax.axhline(y=stats.lidar_distance_mean, color='r', linestyle='--', 
                      label=f'Mean: {stats.lidar_distance_mean:.1f} cm')
            ax.set_xlabel('Time')
            ax.set_ylabel('Distance (cm)')
            ax.set_title('LiDAR Distance Over Time')
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            filepath = output_dir / 'lidar_distance.png'
            plt.savefig(filepath, dpi=150)
            plt.close()
            generated_files.append(filepath)
            print(f"  ✓ {filepath.name}")
    
    # 5. Alert Summary Pie Chart
    if stats.alert_counts:
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = list(stats.alert_counts.keys())
        sizes = list(stats.alert_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax.set_title('Alert Type Distribution')
        plt.tight_layout()
        
        filepath = output_dir / 'alert_distribution.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        generated_files.append(filepath)
        print(f"  ✓ {filepath.name}")
    
    # 6. Summary Dashboard
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # FPS gauge-like display
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(['FPS'], [stats.fps_mean], color='green', height=0.5)
    ax1.set_xlim(0, max(30, stats.fps_max * 1.2))
    ax1.set_title(f'Avg FPS: {stats.fps_mean:.1f}')
    ax1.axvline(x=15, color='orange', linestyle='--', alpha=0.7)
    
    # Lane validity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie([stats.lane_valid_ratio, 1 - stats.lane_valid_ratio], 
           labels=['Valid', 'Invalid'], autopct='%1.1f%%',
           colors=['green', 'red'], startangle=90)
    ax2.set_title('Lane Detection')
    
    # Latency summary
    ax3 = fig.add_subplot(gs[0, 2])
    latencies = ['Lane', 'YOLO', 'Capture']
    values = [stats.lane_latency_mean, stats.yolo_latency_mean, stats.capture_latency_mean]
    colors = ['blue', 'green', 'purple']
    bars = ax3.bar(latencies, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Mean Latency (ms)')
    ax3.set_title('Processing Latencies')
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Session info text
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    info_text = f"""
    SESSION SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Frames: {stats.total_frames:,}    Duration: {stats.duration_seconds/60:.1f} minutes    FPS: {stats.fps_mean:.1f} ± {stats.fps_std:.1f}
    
    Lane Valid: {stats.lane_valid_ratio*100:.1f}%    YOLO Skip Rate: {stats.yolo_skip_ratio*100:.1f}%    Avg Detections: {stats.avg_detections:.2f}
    
    Collision Risk Frames: {stats.collision_risk_frames}    Dropped Frames: {stats.dropped_frames}    IP Reconnects: {stats.ip_reconnects}
    """
    ax4.text(0.5, 0.5, info_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Latency over time mini chart
    ax5 = fig.add_subplot(gs[2, :])
    if 'lane_latency_ms' in df.columns and 'timestamp' in df.columns:
        window = max(1, len(df) // 50)
        df['lane_lat_smooth'] = df['lane_latency_ms'].rolling(window=window, min_periods=1).mean()
        ax5.plot(df['timestamp'], df['lane_lat_smooth'], label='Lane', alpha=0.8)
        if 'yolo_latency_ms' in df.columns:
            df['yolo_lat_smooth'] = df['yolo_latency_ms'].rolling(window=window, min_periods=1).mean()
            ax5.plot(df['timestamp'], df['yolo_lat_smooth'], label='YOLO', alpha=0.8)
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Latency (ms)')
        ax5.set_title('Processing Latency Over Time (Smoothed)')
        ax5.legend()
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.suptitle('Driver Assistant - Telemetry Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    filepath = output_dir / 'dashboard.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    generated_files.append(filepath)
    print(f"  ✓ {filepath.name}")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Driver Assistant telemetry files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/analyze_telemetry.py telemetry.jsonl
    python tools/analyze_telemetry.py telemetry.jsonl --output-dir reports/
    python tools/analyze_telemetry.py data/session1.jsonl -o analysis/
        """
    )
    parser.add_argument('telemetry_file', type=str, nargs='?', default='telemetry.jsonl',
                       help='Path to telemetry JSONL file (default: telemetry.jsonl)')
    parser.add_argument('-o', '--output-dir', type=str, default='telemetry_reports',
                       help='Output directory for graphs (default: telemetry_reports)')
    parser.add_argument('--no-graphs', action='store_true',
                       help='Skip graph generation, print stats only')
    
    args = parser.parse_args()
    
    telemetry_path = Path(args.telemetry_file)
    output_dir = Path(args.output_dir)
    
    if not telemetry_path.exists():
        print(f"Error: Telemetry file not found: {telemetry_path}")
        sys.exit(1)
    
    print(f"\n📁 Loading telemetry from: {telemetry_path}")
    
    try:
        df = load_telemetry(telemetry_path)
        stats = compute_stats(df)
        print_stats(stats)
        
        if not args.no_graphs:
            print(f"\n📈 Generating graphs in: {output_dir}/")
            generated_files = generate_graphs(df, stats, output_dir)
            print(f"\n✅ Generated {len(generated_files)} graph(s)")
            print(f"   Output directory: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
