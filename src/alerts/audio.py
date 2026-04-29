"""Audio alert manager using Python's winsound (Windows) or pygame/beep (Linux/Pi)."""

import platform
import threading
import time
import subprocess
import os
from typing import Optional
from .types import AlertType, AlertEvent


class AudioAlertManager:
    """
    Manages audio alerts using Python built-in sound generation.
    
    On Windows: Uses winsound.Beep() for tone generation
    On Linux/Pi: Uses pygame for tone generation (preferred) or console beep fallback
    
    Features:
    - Non-blocking playback in separate thread
    - Priority preemption (higher priority interrupts lower)
    - Cooldown between same alert type
    """
    
    # Frequency and duration patterns for each alert type
    # Format: [(frequency_hz, duration_ms), ...]
    ALERT_PATTERNS = {
        AlertType.COLLISION_IMMINENT: [
            (2000, 150), (0, 50), (2000, 150), (0, 50), (2000, 150), (0, 50), (2000, 300)
        ],  # Urgent rapid beeps
        AlertType.LANE_DEPARTURE_LEFT: [
            (1000, 200), (0, 100), (1000, 200), (0, 100), (1000, 200)
        ],  # Three medium beeps
        AlertType.LANE_DEPARTURE_RIGHT: [
            (1000, 200), (0, 100), (1000, 200), (0, 100), (1000, 200)
        ],  # Three medium beeps
        AlertType.TRAFFIC_LIGHT_RED: [
            (1500, 400), (0, 100), (1500, 400)
        ],  # Two high warning beeps for red light
        AlertType.TRAFFIC_LIGHT_YELLOW: [
            (1000, 300), (0, 150), (1000, 300)
        ],  # Two medium beeps for yellow caution
        AlertType.TRAFFIC_LIGHT_GREEN: [
            (800, 200), (0, 100), (1200, 200)
        ],  # Ascending tone for green light - distinct from others
        AlertType.STOP_SIGN: [
            (600, 300), (0, 100), (600, 300)
        ],  # Two low beeps
        AlertType.SYSTEM_WARNING: [
            (500, 1000)
        ],  # Single long low beep
    }
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._is_windows = platform.system() == "Windows"
        self._is_linux = platform.system() == "Linux"
        self._playing = False
        self._current_priority = 999
        self._stop_flag = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        
        # Try to import winsound on Windows
        self._winsound = None
        if self._is_windows:
            try:
                import winsound
                self._winsound = winsound
            except ImportError:
                pass
        
        # Try to initialize pygame for Linux/Pi audio
        self._pygame_available = False
        self._pygame = None
        if self._is_linux:
            try:
                import pygame
                pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
                pygame.mixer.init()
                self._pygame = pygame
                self._pygame_available = True
            except Exception:
                pass
    
    def play_alert(self, alert: AlertEvent) -> bool:
        """
        Play audio alert for the given event.
        
        Returns True if alert was started, False if skipped.
        """
        if not self.enabled:
            return False
        
        # Check priority - higher priority (lower number) preempts
        if self._playing and alert.priority >= self._current_priority:
            return False
        
        # Stop current playback if any
        self.stop()
        
        # Start new playback
        self._current_priority = alert.priority
        self._stop_flag.clear()
        self._play_thread = threading.Thread(
            target=self._play_pattern,
            args=(alert.alert_type,),
            daemon=True
        )
        self._playing = True
        self._play_thread.start()
        
        return True
    
    def _play_pattern(self, alert_type: AlertType) -> None:
        """Play the sound pattern for an alert type."""
        pattern = self.ALERT_PATTERNS.get(alert_type, [(800, 200)])
        
        try:
            for freq, duration in pattern:
                if self._stop_flag.is_set():
                    break
                
                if freq == 0:
                    # Silent pause
                    time.sleep(duration / 1000.0)
                elif self._winsound:
                    # Windows beep
                    self._winsound.Beep(freq, duration)
                elif self._pygame_available:
                    # Linux/Pi: Generate tone with pygame
                    self._play_pygame_tone(freq, duration)
                else:
                    # Last resort fallback: try system beep command
                    self._play_system_beep(freq, duration)
        finally:
            self._playing = False
            self._current_priority = 999
    
    def _play_pygame_tone(self, frequency: int, duration_ms: int) -> None:
        """Generate and play a tone using pygame."""
        try:
            import numpy as np
            
            sample_rate = 44100
            duration_s = duration_ms / 1000.0
            n_samples = int(sample_rate * duration_s)
            
            # Generate sine wave
            t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Apply fade in/out to avoid clicks (10ms fade)
            fade_samples = int(sample_rate * 0.01)
            if fade_samples > 0 and n_samples > 2 * fade_samples:
                fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
                fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
                wave[:fade_samples] *= fade_in
                wave[-fade_samples:] *= fade_out
            
            # Convert to 16-bit integer
            wave = (wave * 32767).astype(np.int16)
            
            # Create pygame Sound and play
            sound = self._pygame.sndarray.make_sound(wave)
            sound.play()
            time.sleep(duration_s)
            sound.stop()
        except Exception:
            # If pygame tone fails, use timing only
            time.sleep(duration_ms / 1000.0)
    
    def _play_system_beep(self, frequency: int, duration_ms: int) -> None:
        """Try to play beep using system commands (Linux fallback)."""
        try:
            # Try using 'beep' command if available
            subprocess.run(
                ['beep', '-f', str(frequency), '-l', str(duration_ms)],
                capture_output=True,
                timeout=duration_ms / 1000.0 + 0.5
            )
        except Exception:
            # If beep not available, just wait
            time.sleep(duration_ms / 1000.0)
    
    def stop(self) -> None:
        """Stop current playback."""
        self._stop_flag.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)
        self._playing = False
        self._current_priority = 999
    
    @property
    def is_playing(self) -> bool:
        """Check if currently playing an alert."""
        return self._playing


def beep_alert(alert_type: AlertType) -> None:
    """
    Simple synchronous beep for an alert type.
    Use this for quick testing without threading.
    """
    if platform.system() != "Windows":
        print(f"[ALERT] {alert_type.display_name}")
        return
    
    try:
        import winsound
        patterns = AudioAlertManager.ALERT_PATTERNS.get(alert_type, [(800, 200)])
        for freq, duration in patterns:
            if freq > 0:
                winsound.Beep(freq, duration)
            else:
                time.sleep(duration / 1000.0)
    except Exception:
        print(f"[ALERT] {alert_type.display_name}")
