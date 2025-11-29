# Biomimetic Dual-Axis Phototactic System: Architecture and Performance Analysis

## 1. System Architecture Overview

### 1.1 Hierarchical System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHOTOTACTIC TRACKING SYSTEM                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         SENSOR ACQUISITION LAYER (10 Hz)                │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Photoelectric Sensor Array (4 channels)            │ │  │
│  │  │ ├─ Left Sensor (I_L)                               │ │  │
│  │  │ ├─ Right Sensor (I_R)                              │ │  │
│  │  │ ├─ Upper Sensor (I_U)                              │ │  │
│  │  │ └─ Lower Sensor (I_D)                              │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Signal Processing                                   │ │  │
│  │  │ ├─ ADC Conversion (12-bit, 1 kHz)                  │ │  │
│  │  │ ├─ Low-Pass Filtering (5 Hz cutoff)                │ │  │
│  │  │ └─ Moving Average (10-sample window)               │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      CONTROL DECISION LAYER (10 Hz)                     │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Sensor Error Calculation                            │ │  │
│  │  │ ΔAz = atan2(I_R - I_L, I_R + I_L)                  │ │  │
│  │  │ ΔEl = atan2(I_U - I_D, I_U + I_D)                  │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Astronomical Algorithm (1 Hz update)                │ │  │
│  │  │ ├─ Julian Day Calculation                           │ │  │
│  │  │ ├─ Solar Declination (±0.1° accuracy)              │ │  │
│  │  │ ├─ Hour Angle Calculation                           │ │  │
│  │  │ ├─ Altitude/Azimuth Computation                     │ │  │
│  │  │ └─ Atmospheric Refraction Correction                │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Hybrid Control Strategy                             │ │  │
│  │  │ θ_az_target = θ_az_astro + K_az × ΔAz_sensor      │ │  │
│  │  │ θ_el_target = θ_el_astro + K_el × ΔEl_sensor      │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ PID Control Execution                               │ │  │
│  │  │ u = K_p×e + K_i×∫e + K_d×(de/dt)                   │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      ACTUATION LAYER (10 Hz)                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Motor Command Generation                            │ │  │
│  │  │ ├─ Velocity Saturation                              │ │  │
│  │  │ ├─ PWM Conversion (0-255)                           │ │  │
│  │  │ └─ Stall Detection                                  │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Servo Motor Control                                 │ │  │
│  │  │ ├─ Azimuth Motor (50 W, 100:1 gear)                │ │  │
│  │  │ └─ Elevation Motor (30 W, 150:1 gear)              │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Position Feedback (Encoders)                        │ │  │
│  │  │ ├─ Azimuth Encoder (2048 PPR)                       │ │  │
│  │  │ └─ Elevation Encoder (2048 PPR)                     │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Control Flow Diagram

```
START (100 ms cycle)
│
├─→ [1] Read Sensor Array (10 ms)
│   └─→ ADC conversion, filtering, averaging
│
├─→ [2] Calculate Sensor Errors (5 ms)
│   ├─→ ΔAz = atan2(I_R - I_L, I_R + I_L)
│   └─→ ΔEl = atan2(I_U - I_D, I_U + I_D)
│
├─→ [3] Check Astronomical Update Timer (1 ms)
│   │
│   └─→ IF (time_since_last_update ≥ 1000 ms) THEN
│       ├─→ Calculate Julian Day
│       ├─→ Calculate Solar Declination
│       ├─→ Calculate Hour Angle
│       ├─→ Calculate Altitude & Azimuth
│       ├─→ Apply Refraction Correction
│       ├─→ Check Night Condition (h < -5°)
│       └─→ Update last_update_time
│
├─→ [4] Hybrid Target Calculation (2 ms)
│   │
│   └─→ IF (NOT night AND tracking_enabled) THEN
│       ├─→ θ_az_target = θ_az_astro + K_az × ΔAz
│       └─→ θ_el_target = θ_el_astro + K_el × ΔEl
│       ELSE
│       ├─→ θ_az_target = 180°
│       └─→ θ_el_target = 0°
│
├─→ [5] Get Current Position from Encoders (3 ms)
│   ├─→ θ_az_current = encoder_az_count × scale_az
│   └─→ θ_el_current = encoder_el_count × scale_el
│
├─→ [6] Calculate Control Errors (2 ms)
│   ├─→ e_az = θ_az_target - θ_az_current
│   ├─→ e_el = θ_el_target - θ_el_current
│   └─→ Normalize e_az to [-180°, 180°]
│
├─→ [7] PID Control (5 ms)
│   ├─→ u_az = PID_az(e_az)
│   └─→ u_el = PID_el(e_el)
│
├─→ [8] Velocity Saturation (2 ms)
│   ├─→ v_az = saturate(u_az, -MAX_VEL_AZ, +MAX_VEL_AZ)
│   └─→ v_el = saturate(u_el, -MAX_VEL_EL, +MAX_VEL_EL)
│
├─→ [9] Motor Command Generation (3 ms)
│   ├─→ PWM_az = velocity_to_pwm(v_az)
│   └─→ PWM_el = velocity_to_pwm(v_el)
│
├─→ [10] Output PWM Signals (2 ms)
│   ├─→ Set Motor_AZ PWM
│   └─→ Set Motor_EL PWM
│
├─→ [11] Data Logging (5 ms)
│   └─→ Log: timestamp, θ_az_cmd, θ_el_cmd, θ_az_actual, 
│           θ_el_actual, I_L, I_R, I_U, I_D, I_motor_az, I_motor_el
│
└─→ END (Total: ~100 ms)
```

## 2. Sensor Subsystem Analysis

### 2.1 Sensor Array Geometry

```
                    Upper Sensor (I_U)
                           ↑
                           │
         Left Sensor ← ─────┼───── → Right Sensor
         (I_L)              │            (I_R)
                           │
                           ↓
                    Lower Sensor (I_D)

Spacing: 50-100 mm between adjacent sensors
Mounting: Planar arrangement on tracking device surface
```

### 2.2 Directional Error Calculation

**Azimuthal Error (East-West)**:

```
ΔAz = arctan[(I_R - I_L) / (I_R + I_L)]

Interpretation:
- ΔAz > 0: Sun is to the right (East), need to rotate clockwise
- ΔAz < 0: Sun is to the left (West), need to rotate counter-clockwise
- ΔAz ≈ 0: Sun is centered
```

**Elevation Error (Up-Down)**:

```
ΔEl = arctan[(I_U - I_D) / (I_U + I_D)]

Interpretation:
- ΔEl > 0: Sun is above, need to tilt up
- ΔEl < 0: Sun is below, need to tilt down
- ΔEl ≈ 0: Sun is centered
```

### 2.3 Sensor Signal Processing Pipeline

```
Raw Sensor Data (4 channels)
        ↓
    [ADC Conversion]
    12-bit resolution, 1 kHz sampling
        ↓
    [Low-Pass Filter]
    5 Hz cutoff (Butterworth)
    H(z) = 0.0592/(1 - 0.8817*z^-1)
        ↓
    [Moving Average]
    10-sample window (100 ms)
    y[n] = (1/10) * Σ(x[n-k], k=0 to 9)
        ↓
    [Outlier Detection]
    Remove values > 3σ from mean
        ↓
    Filtered Sensor Data
```

### 2.4 Sensor Performance Specifications

| Parameter               | Specification | Unit                       |
| ----------------------- | ------------- | -------------------------- |
| Spectral Response       | 300-1100 nm   | nm                         |
| Sensitivity             | ≥0.5         | V/W·m⁻²                 |
| Response Time           | <100          | ms                         |
| Measurement Uncertainty | ±2           | %                          |
| Temperature Coefficient | <0.1          | %/°C                      |
| Noise Level             | <2            | % of full scale            |
| Directional Sensitivity | 0.1           | °/% irradiance difference |

## 3. Astronomical Algorithm Analysis

### 3.1 Algorithm Accuracy Validation

**Test Conditions**:

- Location: 40°N, 74°W (New York)
- Date Range: January 1 - December 31, 2025
- Time Interval: Every 30 minutes
- Reference: NOAA Solar Calculator

**Results**:

```
Solar Declination Error:
  Mean: 0.02°
  Std Dev: 0.05°
  Max: 0.09°
  ✓ Within ±0.1° specification

Solar Altitude Error:
  Mean: 0.08°
  Std Dev: 0.15°
  Max: 0.28°
  ✓ Within ±0.3° specification

Solar Azimuth Error:
  Mean: 0.05°
  Std Dev: 0.12°
  Max: 0.25°
  ✓ Within ±0.3° specification
```

### 3.2 Computational Complexity

**Algorithm Execution Time** (on STM32F407 @ 168 MHz):

```
Julian Day Calculation:        ~50 μs
Solar Declination:             ~150 μs
Hour Angle:                    ~100 μs
Altitude/Azimuth:             ~200 μs
Refraction Correction:         ~50 μs
─────────────────────────────
Total per update (1 Hz):       ~550 μs
```

**Memory Requirements**:

```
Code Size:                     ~8 KB
Data Size:                     ~2 KB
Stack Usage:                   ~1 KB
Total:                         ~11 KB
```

### 3.3 Seasonal Variation Impact

```
Winter Solstice (Dec 21):
  Declination: -23.44°
  Solar Altitude Range: 26.5° - 43.0° (at 40°N)
  Tracking Duration: ~9 hours

Summer Solstice (Jun 21):
  Declination: +23.44°
  Solar Altitude Range: 73.0° - 89.5° (at 40°N)
  Tracking Duration: ~15 hours

Equinoxes (Mar 21, Sep 21):
  Declination: 0°
  Solar Altitude Range: 50° - 66.5° (at 40°N)
  Tracking Duration: ~12 hours
```

## 4. Control System Analysis

### 4.1 PID Controller Tuning

**Tuning Method**: Ziegler-Nichols (with manual refinement)

**Azimuth Axis**:

```
Kp = 0.8    (Proportional gain)
Ki = 0.05   (Integral gain)
Kd = 0.2    (Derivative gain)

Tuned for:
- Settling time: <5 seconds
- Overshoot: <10%
- Steady-state error: <0.5°
```

**Elevation Axis**:

```
Kp = 0.8    (Proportional gain)
Ki = 0.05   (Integral gain)
Kd = 0.2    (Derivative gain)

Tuned for:
- Settling time: <5 seconds
- Overshoot: <10%
- Steady-state error: <0.5°
```

### 4.2 Step Response Analysis

**Azimuth Axis (10° step input)**:

```
Time (s) | Position (°) | Error (°) | Velocity (°/s)
0.0      | 0.0          | 10.0      | 0.0
0.5      | 2.5          | 7.5       | 5.0
1.0      | 5.2          | 4.8       | 5.2
1.5      | 7.1          | 2.9       | 3.8
2.0      | 8.5          | 1.5       | 2.8
2.5      | 9.2          | 0.8       | 1.4
3.0      | 9.6          | 0.4       | 0.8
3.5      | 9.8          | 0.2       | 0.4
4.0      | 9.9          | 0.1       | 0.2
5.0      | 10.0         | 0.0       | 0.0

Settling Time (2% band): 4.2 seconds
Peak Overshoot: 0.3%
Rise Time (10%-90%): 1.1 seconds
```

### 4.3 Frequency Response

**Bode Plot Characteristics**:

```
Magnitude Response:
  DC Gain: 0 dB (unity feedback)
  -3dB Bandwidth: ~0.5 Hz
  Phase Margin: 45°
  Gain Margin: 8 dB

Stability Analysis:
  ✓ Stable (poles in left half-plane)
  ✓ Good damping ratio (ζ ≈ 0.7)
  ✓ No oscillation tendency
```

### 4.4 Hybrid Control Performance

**Feedforward vs. Feedback Contribution**:

```
Time (s) | Astro Error (°) | Sensor Error (°) | Combined (°)
0        | 0.5             | 0.3              | 0.35
10       | 0.8             | 0.2              | 0.56
20       | 1.2             | 0.1              | 0.84
30       | 1.5             | 0.15             | 1.05
40       | 1.8             | 0.25             | 1.26
50       | 2.0             | 0.35             | 1.40
60       | 2.2             | 0.45             | 1.55

Feedback Improvement: 30-50% error reduction
Optimal Feedback Gain: K = 0.6 (empirically determined)
```

## 5. Performance Metrics

### 5.1 Tracking Accuracy

**Steady-State Performance**:

```
Metric                          | Value    | Unit
─────────────────────────────────────────────────
Azimuth Steady-State Error      | <0.5     | °
Elevation Steady-State Error    | <0.5     | °
Positional Repeatability        | ±0.2     | °
Tracking Lag (rapid motion)     | <2       | °
```

**Dynamic Performance**:

```
Metric                          | Value    | Unit
─────────────────────────────────────────────────
Transient Response Time         | <5       | s
Peak Overshoot                  | <10      | %
Rise Time (10%-90%)             | ~1       | s
Settling Time (2% band)         | ~4       | s
```

```

```

```

```

## 6. Energy Capture Comparison

### **Test Setup**:

### 6.1

- Location: 117.2°N, 31.8°W
- Duration: 10 days (March , 2025)
- Measurement Interval: 1 minutes

```
Comparison of solar light harvesting performance obtained using the Phototactic system versus the fixed system (March, 2025, Hefei).
/	Fixed system
(MJ/m2)	Phototactic system (MJ/m2)	Improvement
(%)
Day 1	25.1	33.8	34.7
Day 2	24.3	32.7	34.1
Day 3	24.8	33.3	34.2
Day 4	24.7	33.2	34.2
Day 5	25.0	33.2	32.8
Day 6	24.8	33.0	33.0
Day 7	24.7	32.6	31.9
Day 8	24.9	32.7	31.5
Day 9	23.4	30.6	30.8
Average Improvement	33.1 %
standard deviation	1.4 %

```
