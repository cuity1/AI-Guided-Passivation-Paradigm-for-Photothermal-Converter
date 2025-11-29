# Biomimetic Dual-Axis Phototactic System: Implementation and Testing Guide

## 1. Hardware Implementation Guide

### 1.1 PCB Design Specifications

**Main Control Board (MCU Board)**:

```
Dimensions: 100 mm × 80 mm
Layers: 4-layer PCB (2 oz copper)
Trace Width: 
  - Power: 0.5 mm (1 A per 0.5 mm)
  - Signal: 0.2 mm
  - High-speed: 0.15 mm

Component Placement:
  - MCU (STM32F407VG): Center of board
  - ADC (ADS1256): Near sensor connectors
  - Motor drivers: Near motor connectors
  - Power supply: Corner of board
  - Decoupling capacitors: Close to power pins

Power Distribution:
  - 24V input: 2.5 mm² wire
  - 5V regulated: 1.5 mm² wire
  - 3.3V regulated: 1.0 mm² wire
  - Ground plane: Continuous under all components
```

**Sensor Interface Board**:

```
Dimensions: 50 mm × 50 mm
Layers: 2-layer PCB
Components:
  - 4× Photodiode amplifier circuits
  - 4× Low-pass filter networks (5 Hz cutoff)
  - 1× Reference voltage regulator
  - Connector: 5-pin header to main board

Circuit per Channel:
  Photodiode → Transimpedance Amplifier → Low-Pass Filter → ADC
  Gain: 10^6 V/A (adjustable via resistor selection)
```

**Motor Driver Board** (optional separate board):

```
Dimensions: 80 mm × 60 mm
Layers: 2-layer PCB
Components:
  - 2× PWM motor drivers (5 A max)
  - 2× Current sense resistors (0.1 Ω, 1%)
  - 2× Encoder input circuits
  - Connectors: Motor, encoder, PWM control

Thermal Management:
  - Motor drivers on 2 oz copper
  - Thermal vias under driver ICs
  - Aluminum heatsink (if needed)
```

### 1.2 Connector Specifications

**Sensor Connector (5-pin, 2.54 mm pitch)**:

```
Pin 1: +5V (sensor power)
Pin 2: GND
Pin 3: Sensor signal (analog)
Pin 4: GND
Pin 5: +3.3V (reference)
```

**Motor Connector (6-pin, 2.54 mm pitch)**:

```
Pin 1: Motor +24V
Pin 2: Motor GND
Pin 3: Motor PWM (to driver)
Pin 4: Encoder A
Pin 5: Encoder B
Pin 6: Encoder GND
```

**Power Connector (XT60 or Anderson)**:

```
Input: 24V ±10%, 5 A max
Polarity: Red = +24V, Black = GND
Reverse polarity protection: Schottky diode
```

### 1.3 Component Selection Criteria

**Servo Motors**:

```
Selection Criteria:
  - Rated torque > 2× maximum load torque
  - Efficiency > 80%
  - Encoder resolution ≥ 2048 PPR
  - Operating temperature: -10°C to +50°C
  - IP rating: IP54 or better

Recommended Models:
  - Azimuth: Maxon EC-45 flat 50W, 100:1 gearbox
  - Elevation: Maxon EC-32 50W, 150:1 gearbox
```

**Photoelectric Sensors**:

```
Selection Criteria:
  - Spectral response: 300-1100 nm
  - Sensitivity: ≥0.5 V/W·m⁻²
  - Response time: <100 ms
  - Temperature coefficient: <0.1%/°C
  - Linearity: >99%

Recommended Models:
  - Hamamatsu S1133 (silicon photodiode)
  - Hamamatsu S1087 (with integrated amplifier)
```

**Rotary Encoders**:

```
Selection Criteria:
  - Type: Incremental, quadrature output
  - Resolution: 2048 PPR (minimum)
  - Operating temperature: -10°C to +50°C
  - IP rating: IP67 (sealed)
  - Connector: M12 or DT04-2P

Recommended Models:
  - Renishaw OME 2048
  - Heidenhain ERN 1381
```

---

## 2. Software Implementation Guide

### 2.1 Development Environment Setup

**Required Tools**:

```
IDE: STM32CubeIDE (free, Eclipse-based)
Compiler: ARM GCC (included with IDE)
Debugger: ST-Link v2 (USB programmer)
Version Control: Git
Documentation: Doxygen
```

**Project Structure**:

```
phototactic_system/
├── Core/
│   ├── Inc/
│   │   ├── main.h
│   │   ├── phototactic_system.h
│   │   ├── sensor_interface.h
│   │   ├── motor_control.h
│   │   └── astronomical_algorithms.h
│   └── Src/
│       ├── main.c
│       ├── phototactic_system.c
│       ├── sensor_interface.c
│       ├── motor_control.c
│       └── astronomical_algorithms.c
├── Drivers/
│   ├── STM32F4xx_HAL_Driver/
│   └── CMSIS/
├── Middleware/
│   └── FreeRTOS/
├── Documentation/
│   ├── design_document.md
│   ├── api_reference.md
│   └── testing_procedures.md
├── Tests/
│   ├── unit_tests/
│   ├── integration_tests/
│   └── system_tests/
└── CMakeLists.txt
```

### 2.2 Firmware Development Workflow

**Step 1: Initialize MCU Peripherals**

```c
// main.c
void SystemInit_Custom(void) {
    // Configure system clock (168 MHz)
    SystemClock_Config();
  
    // Initialize GPIO
    GPIO_Init();
  
    // Initialize ADC (12-bit, 1 kHz)
    ADC_Init();
  
    // Initialize PWM timers
    PWM_Init();
  
    // Initialize UART (for debugging)
    UART_Init(115200);
  
    // Initialize SPI/I2C (if needed)
    SPI_Init();
  
    // Start FreeRTOS scheduler
    osKernelStart();
}
```

**Step 2: Implement Sensor Interface**

```c
// sensor_interface.c
void SensorTask(void *argument) {
    sensor_data_t sensor_data;
  
    while (1) {
        // Read all 4 ADC channels
        sensor_data.I_left = ADC_Read(ADC_CH_LEFT);
        sensor_data.I_right = ADC_Read(ADC_CH_RIGHT);
        sensor_data.I_upper = ADC_Read(ADC_CH_UPPER);
        sensor_data.I_lower = ADC_Read(ADC_CH_LOWER);
        sensor_data.timestamp = osKernelGetTickCount();
      
        // Apply filtering
        ApplySensorFilter(&sensor_data);
      
        // Send to control task
        osMessageQueuePut(sensorQueueHandle, &sensor_data, 0, 0);
      
        // Wait 100 ms (10 Hz)
        osDelay(100);
    }
}
```

**Step 3: Implement Control Loop**

```c
// phototactic_system.c
void ControlTask(void *argument) {
    sensor_data_t sensor_data;
    control_output_t output;
  
    while (1) {
        // Wait for sensor data
        if (osMessageQueueGet(sensorQueueHandle, &sensor_data, NULL, 100) == osOK) {
            // Execute control step
            phototactic_control_step(&sensor_data, &current_datetime, &output);
          
            // Output motor commands
            SetMotorPWM(MOTOR_AZ, output.pwm_azimuth);
            SetMotorPWM(MOTOR_EL, output.pwm_elevation);
          
            // Log data
            LogSystemState(&sensor_data, &output);
        }
    }
}
```

### 2.3 Real-Time Operating System Configuration

**FreeRTOS Configuration** (FreeRTOSConfig.h):

```c
#define configUSE_PREEMPTION                1
#define configCPU_CLOCK_HZ                  168000000
#define configTICK_RATE_HZ                  1000
#define configMAX_PRIORITIES                5
#define configMINIMAL_STACK_SIZE            128
#define configTOTAL_HEAP_SIZE               40960

// Task priorities (higher number = higher priority)
#define PRIORITY_SENSOR_TASK                3
#define PRIORITY_CONTROL_TASK               4  // Highest
#define PRIORITY_LOGGING_TASK               2
#define PRIORITY_ASTRO_TASK                 1
```

**Task Configuration**:

```c
// Create tasks
osThreadNew(SensorTask, NULL, &sensor_task_attr);      // 10 Hz
osThreadNew(ControlTask, NULL, &control_task_attr);    // 10 Hz
osThreadNew(AstronomyTask, NULL, &astro_task_attr);    // 1 Hz
osThreadNew(LoggingTask, NULL, &logging_task_attr);    // 1 Hz

// Task stack sizes
#define SENSOR_STACK_SIZE    256
#define CONTROL_STACK_SIZE   512
#define ASTRO_STACK_SIZE     512
#define LOGGING_STACK_SIZE   256
```

### 2.4 Code Quality Standards

**Coding Style**:

```c
// Use consistent naming conventions
// Variables: snake_case
float solar_altitude;
float motor_current_az;

// Functions: snake_case with module prefix
float phototactic_calculate_error(float target, float actual);
void motor_control_set_pwm(uint8_t pwm_value);

// Constants: UPPER_CASE
#define MAX_MOTOR_SPEED_DPS    18.0f
#define SENSOR_FILTER_CUTOFF   5.0f

// Macros: Use inline functions when possible
static inline float saturate(float val, float min, float max) {
    return (val > max) ? max : (val < min) ? min : val;
}
```

**Documentation Standards**:

```c
/**
 * Calculate solar position for given location and time
 * 
 * This function implements the algorithm described in Meeus (1998)
 * with atmospheric refraction correction per NOAA standards.
 * 
 * @param location Pointer to observer location (latitude, longitude)
 * @param datetime Pointer to date/time in UTC
 * @param position Pointer to output solar position structure
 * 
 * @return 0 on success, negative on error
 * 
 * @note Accuracy: ±0.3° for altitude and azimuth
 * @note Computation time: ~550 μs on STM32F407
 * 
 * @see calculate_solar_declination()
 * @see calculate_hour_angle()
 */
int phototactic_calculate_solar_position(
    const phototactic_location_t *location,
    const phototactic_datetime_t *datetime,
    phototactic_solar_position_t *position);
```

---

## 3. Testing Procedures

### 3.1 Unit Testing

**Test Framework**: Unity (C unit testing framework)

**Sensor Processing Tests**:

```c
void test_calculate_azimuth_error_positive(void) {
    sensor_data_t sensor_data = {
        .I_left = 100.0f,
        .I_right = 110.0f,
        .I_upper = 105.0f,
        .I_lower = 105.0f
    };
  
    float error = calculate_azimuth_error(&sensor_data);
  
    TEST_ASSERT_GREATER_THAN(0.0f, error);  // Positive error
    TEST_ASSERT_LESS_THAN(1.0f, error);     // Reasonable magnitude
}

void test_calculate_azimuth_error_zero(void) {
    sensor_data_t sensor_data = {
        .I_left = 100.0f,
        .I_right = 100.0f,
        .I_upper = 100.0f,
        .I_lower = 100.0f
    };
  
    float error = calculate_azimuth_error(&sensor_data);
  
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, error);
}
```

**Astronomical Algorithm Tests**:

```c
void test_solar_declination_summer_solstice(void) {
    // June 21, 2025
    datetime_t dt = {2025, 6, 21, 12, 0, 0};
    float jd = calculate_julian_day(&dt);
    float declination = calculate_solar_declination(jd);
  
    // Expected: ~23.44°
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 23.44f, declination);
}

void test_solar_declination_winter_solstice(void) {
    // December 21, 2025
    datetime_t dt = {2025, 12, 21, 12, 0, 0};
    float jd = calculate_julian_day(&dt);
    float declination = calculate_solar_declination(jd);
  
    // Expected: ~-23.44°
    TEST_ASSERT_FLOAT_WITHIN(0.1f, -23.44f, declination);
}
```

**PID Controller Tests**:

```c
void test_pid_controller_step_response(void) {
    pid_controller_t pid;
    pid_init(&pid, 0.8f, 0.05f, 0.2f, 18.0f, -18.0f);
  
    float error = 10.0f;  // 10° error
    float output = pid_update(&pid, error, 0.1f);
  
    // First step should produce proportional response
    float expected = 0.8f * 10.0f;  // Kp * error
    TEST_ASSERT_FLOAT_WITHIN(1.0f, expected, output);
}

void test_pid_controller_integral_windup(void) {
    pid_controller_t pid;
    pid_init(&pid, 0.8f, 0.05f, 0.2f, 10.0f, -10.0f);
  
    // Apply constant error for long time
    for (int i = 0; i < 100; i++) {
        float output = pid_update(&pid, 5.0f, 0.1f);
        TEST_ASSERT_LESS_THAN_OR_EQUAL(10.0f, output);
        TEST_ASSERT_GREATER_THAN_OR_EQUAL(-10.0f, output);
    }
}
```

### 3.2 Integration Testing

**Sensor-Control Integration**:

```c
void test_sensor_to_motor_integration(void) {
    // Initialize system
    control_system_t sys;
    control_system_init(&sys, 40.0f, -74.0f, -5);
  
    // Simulate sensor data (sun to the right)
    sensor_data_t sensor_data = {
        .I_left = 80.0f,
        .I_right = 120.0f,
        .I_upper = 100.0f,
        .I_lower = 100.0f,
        .timestamp = 0
    };
  
    // Execute control step
    float pwm_az, pwm_el;
    control_step(&sys, &sensor_data, &pwm_az, &pwm_el);
  
    // Should produce positive azimuth command (turn right)
    TEST_ASSERT_GREATER_THAN(127.5f, pwm_az);
}
```

**Astronomical-Control Integration**:

```c
void test_astronomical_tracking(void) {
    control_system_t sys;
    control_system_init(&sys, 40.0f, -74.0f, -5);
  
    // Set to noon on summer solstice
    sys.datetime = {2025, 6, 21, 12, 0, 0};
    sys.current_time = 0;
  
    // Execute multiple control steps
    for (int i = 0; i < 10; i++) {
        sensor_data_t sensor_data = {100, 100, 100, 100, sys.current_time};
        float pwm_az, pwm_el;
      
        control_step(&sys, &sensor_data, &pwm_az, &pwm_el);
      
        // Elevation should be high (sun high in sky at summer solstice)
        TEST_ASSERT_GREATER_THAN(60.0f, sys.theta_el_target);
      
        sys.current_time += 100;
    }
}
```

### 3.3 System Testing

**Bench Testing Setup**:

```
┌─────────────────────────────────────────┐
│     Artificial Light Source             │
│  (Adjustable position, known spectrum)  │
└────────────┬────────────────────────────┘
             │
             ↓
    ┌────────────────────┐
    │ Phototactic System │
    │ (Under Test)       │
    └────────┬───────────┘
             │
    ┌────────┴──────────────────────────┐
    │                                   │
    ↓                                   ↓
┌─────────────────┐          ┌──────────────────┐
│ Position Sensor │          │ Data Logger      │
│ (Theodolite)    │          │ (PC with USB)    │
└─────────────────┘          └──────────────────┘
```

**Test Procedure**:

```
1. Initialization Test
   - Power on system
   - Verify all LEDs illuminate
   - Check serial output for initialization messages
   - Confirm encoder readings are zero

2. Sensor Calibration Test
   - Place under uniform light
   - Record baseline sensor readings
   - Verify all sensors read within ±5% of each other
   - Store calibration coefficients

3. Motor Response Test
   - Command 10° azimuth step
   - Measure response time (should be <5 sec)
   - Verify no overshoot (should be <10%)
   - Repeat for elevation axis

4. Tracking Accuracy Test
   - Position light source at known angle
   - Record system response
   - Measure steady-state error
   - Repeat at 10° intervals across full range

5. Diurnal Tracking Test
   - Simulate sun motion over 24 hours
   - Use motorized light source on predefined path
   - Record tracking error every 10 minutes
   - Verify mean error <0.5°

6. Environmental Test
   - Temperature: -10°C to +50°C
   - Humidity: 10-90% RH
   - Vibration: 2 g at 10-500 Hz
   - Verify performance within specifications
```

### 3.4 Field Testing

**Outdoor Validation**:

```
Test Duration: 30 days
Location: 40°N, 74°W (New York)
Measurement Interval: 10 minutes

Equipment:
  - Reference pyranometer (±2% accuracy)
  - Theodolite (±0.1° accuracy)
  - Data logger (1 Hz sampling)
  - Weather station (temperature, humidity, wind)

Metrics Collected:
  - Solar altitude (predicted vs. actual)
  - Solar azimuth (predicted vs. actual)
  - Tracking error (system vs. reference)
  - Irradiance on tracking device
  - Irradiance on fixed reference
  - System power consumption
  - Motor current draw
  - Ambient temperature
  - Cloud cover (visual estimation)

Data Analysis:
  - Mean tracking error
  - Standard deviation
  - 95th percentile error
  - Energy capture improvement
  - Seasonal variation
  - Weather impact
```

---

## 4. Commissioning Checklist

### 4.1 Pre-Deployment Verification

```
□ Hardware Assembly
  □ All components soldered correctly
  □ No cold solder joints
  □ No shorts or open circuits
  □ Connectors properly seated
  □ Mechanical alignment verified

□ Firmware Installation
  □ Code compiles without errors
  □ No compiler warnings
  □ All unit tests pass
  □ Integration tests pass
  □ Firmware flashed to MCU
  □ Bootloader verified

□ Sensor Calibration
  □ All 4 sensors respond to light
  □ Sensor readings within expected range
  □ Calibration coefficients stored
  □ Temperature compensation verified

□ Motor Testing
  □ Azimuth motor rotates smoothly
  □ Elevation motor rotates smoothly
  □ No grinding or unusual noise
  □ Encoders read correctly
  □ Motor current within limits

□ Control System
  □ PID gains tuned
  □ Step response acceptable
  □ No oscillation
  □ Steady-state error <0.5°

□ Safety Systems
  □ Emergency stop functional
  □ Stall detection working
  □ Current limiting active
  □ Thermal shutdown tested
  □ Watchdog timer functional

□ Documentation
  □ System manual complete
  □ Calibration data recorded
  □ Firmware version documented
  □ Serial number recorded
  □ Warranty information provided
```

### 4.2 Deployment Procedure

```
1. Site Preparation
   - Clear area around installation
   - Verify structural support
   - Check for obstructions
   - Measure GPS coordinates
   - Record local magnetic declination

2. Installation
   - Mount tracking device securely
   - Install sensor array
   - Connect motor cables
   - Connect power supply
   - Ground all metal parts

3. Initialization
   - Power on system
   - Verify startup sequence
   - Confirm time synchronization
   - Set location coordinates
   - Set UTC offset

4. Calibration
   - Perform sensor calibration
   - Verify astronomical calculations
   - Test motor response
   - Confirm tracking accuracy

5. Commissioning
   - Monitor system for 24 hours
   - Verify diurnal tracking
   - Check power consumption
   - Validate data logging
   - Document baseline performance

6. Handover
   - Provide user manual
   - Provide maintenance schedule
   - Provide emergency contacts
   - Train operators
   - Establish support agreement
```

---

## 5. Maintenance and Troubleshooting

### 5.1 Maintenance Schedule

```
Daily:
  - Visual inspection for damage
  - Check for unusual noise
  - Verify LED indicators

Weekly:
  - Clean sensor array (remove dust)
  - Check motor current draw
  - Verify tracking accuracy

Monthly:
  - Inspect mechanical components
  - Check bearing lubrication
  - Verify encoder function
  - Review system logs

Quarterly:
  - Recalibrate sensors
  - Check motor performance
  - Verify astronomical calculations
  - Update firmware if available

Semi-Annually:
  - Regrease bearings
  - Inspect wiring for corrosion
  - Check connector integrity
  - Perform full system test

Annually:
  - Complete system overhaul
  - Replace worn components
  - Recalibrate all sensors
  - Update documentation
```

### 5.2 Troubleshooting Guide

**Problem: System not tracking**

```
Possible Causes:
  1. Tracking disabled in software
     → Solution: Enable tracking via control interface
  
  2. Night time (solar altitude < -5°)
     → Solution: Wait for sunrise
  
  3. Sensor failure
     → Solution: Check sensor readings, recalibrate
  
  4. Motor failure
     → Solution: Check motor current, verify encoder
  
  5. Astronomical algorithm error
     → Solution: Verify date/time/location settings

Diagnostic Steps:
  1. Check system state via serial interface
  2. Verify sensor readings are reasonable
  3. Verify motor commands are being generated
  4. Check motor current draw
  5. Verify encoder position changes
```

**Problem: Large tracking error (>2°)**

```
Possible Causes:
  1. Sensor calibration drift
     → Solution: Recalibrate sensors
  
  2. PID gains incorrect
     → Solution: Re-tune PID controller
  
  3. Mechanical backlash
     → Solution: Inspect gearbox, adjust if needed
  
  4. Encoder misalignment
     → Solution: Realign encoder with motor shaft
  
  5. Astronomical algorithm error
     → Solution: Verify location/time accuracy

Diagnostic Steps:
  1. Compare predicted vs. actual solar position
  2. Check sensor error signals
  3. Verify motor response to commands
  4. Measure mechanical backlash
  5. Recalibrate entire system
```

**Problem: High power consumption (>30W)**

```
Possible Causes:
  1. Motors running continuously
     → Solution: Check for stall condition
  
  2. Bearing friction increased
     → Solution: Regrease bearings
  
  3. Motor efficiency degraded
     → Solution: Replace motor
  
  4. Control loop oscillating
     → Solution: Re-tune PID gains

Diagnostic Steps:
  1. Monitor motor current over time
  2. Check for oscillation in position
  3. Measure bearing resistance
  4. Verify PID controller stability
```

---

## 6. Performance Monitoring

### 6.1 Key Performance Indicators (KPIs)

```
Real-Time Monitoring:
  - Current tracking error: ±0.3° (target)
  - Motor current: 0.5-2.0 A (normal range)
  - System temperature: <50°C
  - Sensor readings: 50-200 V (typical)

Daily Monitoring:
  - Mean tracking error: <0.5°
  - Energy capture: Compare to forecast
  - System uptime: >99%
  - Power consumption: 10-20 W (average)

Weekly Monitoring:
  - Tracking accuracy trend
  - Energy efficiency trend
  - Maintenance needs
  - Firmware updates available

Monthly Monitoring:
  - System MTBF estimate
  - Predictive maintenance alerts
  - Performance vs. specification
  - Seasonal performance variation
```

### 6.2 Data Logging Format

**CSV Log File**:

```
timestamp_ms,az_cmd_deg,el_cmd_deg,az_actual_deg,el_actual_deg,
I_left_V,I_right_V,I_upper_V,I_lower_V,motor_current_az_A,motor_current_el_A

0,180.0,0.0,0.0,0.0,1.5,1.5,1.5,1.5,0.0,0.0
100,180.5,0.2,0.1,0.05,1.6,1.7,1.55,1.45,0.3,0.1
200,181.0,0.4,0.3,0.15,1.8,1.9,1.6,1.4,0.5,0.2
...
```

---
