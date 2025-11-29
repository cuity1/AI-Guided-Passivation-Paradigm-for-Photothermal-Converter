/**
 * Biomimetic Dual-Axis Phototactic System Implementation
 * 
 * This file contains the core implementation of the solar tracking system
 * including sensor processing, astronomical algorithms, and motor control.
 * 
 * Author: Research Team
 * Date: November 2025
 * Version: 1.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

/* ============================================================================
   CONSTANTS AND CONFIGURATION
   ============================================================================ */

#define PI                  3.14159265359f
#define DEG_TO_RAD          (PI / 180.0f)
#define RAD_TO_DEG          (180.0f / PI)

// System configuration
#define CONTROL_FREQUENCY   10      // Hz
#define CONTROL_PERIOD_MS   100     // milliseconds
#define ASTRO_UPDATE_FREQ   1       // Hz
#define ASTRO_UPDATE_MS     1000    // milliseconds

// Motor specifications
#define MAX_VELOCITY_AZ     18.0f   // degrees per second
#define MAX_VELOCITY_EL     12.0f   // degrees per second
#define MAX_ACCELERATION    5.0f    // degrees per second squared

// Sensor array configuration
#define SENSOR_COUNT        4
#define SENSOR_SAMPLING_HZ  10
#define SENSOR_FILTER_CUTOFF 5.0f   // Hz

// PID controller gains
#define KP_AZ               0.8f
#define KI_AZ               0.05f
#define KD_AZ               0.2f

#define KP_EL               0.8f
#define KI_EL               0.05f
#define KD_EL               0.2f

// Feedback gains
#define K_AZ_FEEDBACK       0.6f
#define K_EL_FEEDBACK       0.6f

/* ============================================================================
   DATA STRUCTURES
   ============================================================================ */

typedef struct {
    float I_left;       // Irradiance at left sensor
    float I_right;      // Irradiance at right sensor
    float I_upper;      // Irradiance at upper sensor
    float I_lower;      // Irradiance at lower sensor
    uint32_t timestamp; // Measurement timestamp (ms)
} sensor_data_t;

typedef struct {
    float azimuth;      // Solar azimuth angle (degrees)
    float altitude;     // Solar altitude angle (degrees)
    float declination;  // Solar declination (degrees)
    float hour_angle;   // Hour angle (degrees)
} solar_position_t;

typedef struct {
    float Kp;           // Proportional gain
    float Ki;           // Integral gain
    float Kd;           // Derivative gain
    float integral;     // Accumulated integral term
    float prev_error;   // Previous error for derivative calculation
    float output_max;   // Maximum output saturation
    float output_min;   // Minimum output saturation
} pid_controller_t;

typedef struct {
    float latitude;     // Observer latitude (degrees)
    float longitude;    // Observer longitude (degrees)
    int32_t utc_offset; // UTC offset (hours)
} location_t;

typedef struct {
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
} datetime_t;

typedef struct {
    float theta_az;     // Azimuth angle (degrees)
    float theta_el;     // Elevation angle (degrees)
    float v_az;         // Azimuth velocity (degrees/second)
    float v_el;         // Elevation velocity (degrees/second)
} system_state_t;

/* ============================================================================
   UTILITY FUNCTIONS
   ============================================================================ */

/**
 * Convert degrees to radians
 */
static inline float deg_to_rad(float degrees) {
    return degrees * DEG_TO_RAD;
}

/**
 * Convert radians to degrees
 */
static inline float rad_to_deg(float radians) {
    return radians * RAD_TO_DEG;
}

/**
 * Sine function for degree input
 */
static inline float sin_deg(float degrees) {
    return sinf(deg_to_rad(degrees));
}

/**
 * Cosine function for degree input
 */
static inline float cos_deg(float degrees) {
    return cosf(deg_to_rad(degrees));
}

/**
 * Tangent function for degree input
 */
static inline float tan_deg(float degrees) {
    return tanf(deg_to_rad(degrees));
}

/**
 * Arctangent function returning degrees
 */
static inline float atan_deg(float y, float x) {
    return rad_to_deg(atan2f(y, x));
}

/**
 * Arcsine function returning degrees
 */
static inline float asin_deg(float x) {
    return rad_to_deg(asinf(x));
}

/**
 * Arccosine function returning degrees
 */
static inline float acos_deg(float x) {
    return rad_to_deg(acosf(x));
}

/**
 * Saturate value between min and max
 */
static inline float saturate(float value, float min, float max) {
    if (value > max) return max;
    if (value < min) return min;
    return value;
}

/**
 * Normalize angle to [-180, 180] range
 */
static inline float normalize_angle(float angle) {
    while (angle > 180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
    return angle;
}

/**
 * Low-pass filter (first-order IIR)
 * y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
 */
static float low_pass_filter(float input, float *prev_output, 
                             float cutoff_freq, float sample_freq) {
    float dt = 1.0f / sample_freq;
    float alpha = cutoff_freq * dt / (1.0f + cutoff_freq * dt);
    float output = alpha * input + (1.0f - alpha) * (*prev_output);
    *prev_output = output;
    return output;
}

/* ============================================================================
   SENSOR PROCESSING
   ============================================================================ */

/**
 * Calculate directional error from sensor array
 * Returns error in degrees (positive = need to move right/up)
 */
static float calculate_azimuth_error(const sensor_data_t *sensor_data) {
    float total = sensor_data->I_left + sensor_data->I_right;
    if (total < 1e-6f) return 0.0f;
    
    float ratio = (sensor_data->I_right - sensor_data->I_left) / total;
    // Limit ratio to [-1, 1] for numerical stability
    ratio = saturate(ratio, -1.0f, 1.0f);
    
    // Convert ratio to angle error (approximately 0.1 degree per 1% difference)
    float error_deg = atanf(ratio) * RAD_TO_DEG;
    return error_deg;
}

/**
 * Calculate elevation error from sensor array
 * Returns error in degrees (positive = need to move up)
 */
static float calculate_elevation_error(const sensor_data_t *sensor_data) {
    float total = sensor_data->I_upper + sensor_data->I_lower;
    if (total < 1e-6f) return 0.0f;
    
    float ratio = (sensor_data->I_upper - sensor_data->I_lower) / total;
    // Limit ratio to [-1, 1] for numerical stability
    ratio = saturate(ratio, -1.0f, 1.0f);
    
    // Convert ratio to angle error
    float error_deg = atanf(ratio) * RAD_TO_DEG;
    return error_deg;
}

/**
 * Apply moving average filter to sensor data
 */
static void apply_sensor_filter(sensor_data_t *current, 
                                const sensor_data_t *new_data,
                                float alpha) {
    current->I_left = alpha * new_data->I_left + (1.0f - alpha) * current->I_left;
    current->I_right = alpha * new_data->I_right + (1.0f - alpha) * current->I_right;
    current->I_upper = alpha * new_data->I_upper + (1.0f - alpha) * current->I_upper;
    current->I_lower = alpha * new_data->I_lower + (1.0f - alpha) * current->I_lower;
    current->timestamp = new_data->timestamp;
}

/* ============================================================================
   ASTRONOMICAL ALGORITHMS
   ============================================================================ */

/**
 * Calculate Julian Day Number from calendar date and time
 * Reference: Meeus, Astronomical Algorithms
 */
static float calculate_julian_day(const datetime_t *dt) {
    int32_t a = (14 - dt->month) / 12;
    int32_t y = dt->year + 4800 - a;
    int32_t m = dt->month + 12 * a - 3;
    
    int32_t jdn = dt->day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    
    float jd = jdn + (dt->hour - 12.0f) / 24.0f + dt->minute / 1440.0f + dt->second / 86400.0f;
    
    return jd;
}

/**
 * Calculate solar declination angle
 * Accuracy: ±0.1 degrees
 */
static float calculate_solar_declination(float julian_day) {
    float T = (julian_day - 2451545.0f) / 36525.0f;  // Julian centuries from J2000.0
    
    // Mean longitude of sun
    float L0 = 280.46646f + 36000.76983f * T + 0.0003032f * T * T;
    L0 = fmodf(L0, 360.0f);
    
    // Mean anomaly
    float M = 357.52911f + 35999.05029f * T - 0.0001536f * T * T;
    M = deg_to_rad(M);
    
    // Equation of center
    float C = (1.914602f - 0.004817f * T - 0.000014f * T * T) * sinf(M)
            + (0.019993f - 0.000101f * T) * sinf(2.0f * M)
            + 0.000029f * sinf(3.0f * M);
    
    // Apparent longitude
    float lambda = L0 + C;
    lambda = deg_to_rad(lambda);
    
    // Obliquity of ecliptic
    float epsilon = 23.439291f - 0.0130042f * T - 0.00000016f * T * T + 0.000000504f * T * T * T;
    epsilon = deg_to_rad(epsilon);
    
    // Solar declination
    float declination = asinf(sinf(epsilon) * sinf(lambda));
    
    return rad_to_deg(declination);
}

/**
 * Calculate equation of time (in minutes)
 * Used for precise solar time calculation
 */
static float calculate_equation_of_time(float julian_day) {
    float T = (julian_day - 2451545.0f) / 36525.0f;
    
    float L0 = 280.46646f + 36000.76983f * T + 0.0003032f * T * T;
    float M = 357.52911f + 35999.05029f * T - 0.0001536f * T * T;
    float e = 0.016708634f - 0.000042037f * T - 0.0000001267f * T * T;
    
    float y = tanf(deg_to_rad(L0 / 2.0f));
    y = y * y;
    
    float sin_2L0 = sinf(deg_to_rad(2.0f * L0));
    float sin_M = sinf(deg_to_rad(M));
    float cos_2L0 = cosf(deg_to_rad(2.0f * L0));
    float sin_4L0 = sinf(deg_to_rad(4.0f * L0));
    float sin_2M = sinf(deg_to_rad(2.0f * M));
    
    float eot = 229.18f * (y * sin_2L0 - 2.0f * e * sin_M + 4.0f * e * y * sin_M * cos_2L0
                          - 0.5f * y * y * sin_4L0 - 1.25f * e * e * sin_2M);
    
    return eot;
}

/**
 * Calculate solar altitude and azimuth angles
 * Inputs: latitude, longitude (degrees), datetime, declination
 * Outputs: altitude, azimuth (degrees)
 */
static void calculate_solar_position(const location_t *location,
                                     const datetime_t *dt,
                                     float declination,
                                     solar_position_t *position) {
    // Calculate Julian Day
    float jd = calculate_julian_day(dt);
    
    // Calculate Greenwich Mean Sidereal Time (GMST)
    float T_ut = (jd - 2451545.0f) / 36525.0f;
    float gmst = 280.46061837f + 360.98564724f * (jd - 2451545.0f) 
               + 0.000387933f * T_ut * T_ut - T_ut * T_ut * T_ut / 38710000.0f;
    gmst = fmodf(gmst, 360.0f);
    
    // Calculate Local Sidereal Time (LST)
    float lst = gmst + location->longitude;
    lst = fmodf(lst, 360.0f);
    
    // Calculate Hour Angle
    float hour_angle = lst - 0.0f;  // 0.0 is right ascension of sun (simplified)
    hour_angle = normalize_angle(hour_angle);
    
    // Convert to radians
    float lat_rad = deg_to_rad(location->latitude);
    float dec_rad = deg_to_rad(declination);
    float ha_rad = deg_to_rad(hour_angle);
    
    // Calculate altitude angle
    float sin_alt = sinf(lat_rad) * sinf(dec_rad) 
                  + cosf(lat_rad) * cosf(dec_rad) * cosf(ha_rad);
    sin_alt = saturate(sin_alt, -1.0f, 1.0f);
    float altitude = asinf(sin_alt);
    
    // Calculate azimuth angle
    float cos_alt = cosf(altitude);
    float sin_az = -cosf(dec_rad) * sinf(ha_rad) / cos_alt;
    float cos_az = (sinf(dec_rad) - sinf(lat_rad) * sinf(altitude)) / (cosf(lat_rad) * cos_alt);
    
    sin_az = saturate(sin_az, -1.0f, 1.0f);
    cos_az = saturate(cos_az, -1.0f, 1.0f);
    
    float azimuth = atan2f(sin_az, cos_az);
    
    // Atmospheric refraction correction
    float alt_deg = rad_to_deg(altitude);
    if (alt_deg > -0.833f) {  // Only apply correction above horizon
        float R = 0.96422f / tanf(deg_to_rad(alt_deg + 10.3f / (alt_deg + 5.11f)));
        altitude += deg_to_rad(R / 60.0f);
    }
    
    position->azimuth = rad_to_deg(azimuth);
    position->altitude = rad_to_deg(altitude);
    position->declination = declination;
    position->hour_angle = hour_angle;
    
    // Normalize azimuth to [0, 360)
    if (position->azimuth < 0.0f) position->azimuth += 360.0f;
}

/**
 * Complete solar position calculation
 * This is the main function to call for astronomical calculations
 */
static void calculate_solar_position_complete(const location_t *location,
                                              const datetime_t *dt,
                                              solar_position_t *position) {
    float jd = calculate_julian_day(dt);
    float declination = calculate_solar_declination(jd);
    calculate_solar_position(location, dt, declination, position);
}

/* ============================================================================
   PID CONTROLLER
   ============================================================================ */

/**
 * Initialize PID controller
 */
static void pid_init(pid_controller_t *pid, float Kp, float Ki, float Kd,
                     float output_max, float output_min) {
    pid->Kp = Kp;
    pid->Ki = Ki;
    pid->Kd = Kd;
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
    pid->output_max = output_max;
    pid->output_min = output_min;
}

/**
 * Reset PID controller state
 */
static void pid_reset(pid_controller_t *pid) {
    pid->integral = 0.0f;
    pid->prev_error = 0.0f;
}

/**
 * Execute PID control step
 * dt: time step in seconds
 */
static float pid_update(pid_controller_t *pid, float error, float dt) {
    // Proportional term
    float p_term = pid->Kp * error;
    
    // Integral term with anti-windup
    pid->integral += error * dt;
    float integral_max = pid->output_max / pid->Ki;
    float integral_min = pid->output_min / pid->Ki;
    pid->integral = saturate(pid->integral, integral_min, integral_max);
    float i_term = pid->Ki * pid->integral;
    
    // Derivative term
    float d_term = 0.0f;
    if (dt > 1e-6f) {
        d_term = pid->Kd * (error - pid->prev_error) / dt;
    }
    pid->prev_error = error;
    
    // Combine terms and saturate
    float output = p_term + i_term + d_term;
    output = saturate(output, pid->output_min, pid->output_max);
    
    return output;
}

/* ============================================================================
   MOTOR CONTROL
   ============================================================================ */

/**
 * Convert velocity (degrees/second) to PWM value (0-255)
 */
static uint8_t velocity_to_pwm(float velocity, float max_velocity) {
    // Normalize velocity to [-1, 1]
    float normalized = saturate(velocity / max_velocity, -1.0f, 1.0f);
    
    // Convert to PWM: 127 = stop, 0-127 = reverse, 128-255 = forward
    uint8_t pwm = (uint8_t)(127.5f + normalized * 127.5f);
    
    return pwm;
}

/**
 * Convert PWM value to velocity (degrees/second)
 */
static float pwm_to_velocity(uint8_t pwm, float max_velocity) {
    // Convert PWM to normalized velocity [-1, 1]
    float normalized = (pwm - 127.5f) / 127.5f;
    normalized = saturate(normalized, -1.0f, 1.0f);
    
    // Scale to maximum velocity
    float velocity = normalized * max_velocity;
    
    return velocity;
}

/* ============================================================================
   MAIN CONTROL ALGORITHM
   ============================================================================ */

/**
 * Main control loop function
 * Should be called at CONTROL_FREQUENCY (10 Hz)
 */
typedef struct {
    // Current state
    float theta_az_current;
    float theta_el_current;
    
    // Target positions
    float theta_az_target;
    float theta_el_target;
    
    // Astronomical positions
    float theta_az_astro;
    float theta_el_astro;
    
    // PID controllers
    pid_controller_t pid_az;
    pid_controller_t pid_el;
    
    // Sensor data
    sensor_data_t sensor_data;
    
    // Location and time
    location_t location;
    datetime_t datetime;
    
    // Timing
    uint32_t last_astro_update;
    uint32_t current_time;
    
    // System state
    bool is_tracking;
    bool is_night;
} control_system_t;

/**
 * Initialize control system
 */
static void control_system_init(control_system_t *sys,
                                float latitude, float longitude,
                                int32_t utc_offset) {
    sys->theta_az_current = 0.0f;
    sys->theta_el_current = 0.0f;
    sys->theta_az_target = 0.0f;
    sys->theta_el_target = 0.0f;
    sys->theta_az_astro = 0.0f;
    sys->theta_el_astro = 0.0f;
    
    pid_init(&sys->pid_az, KP_AZ, KI_AZ, KD_AZ, MAX_VELOCITY_AZ, -MAX_VELOCITY_AZ);
    pid_init(&sys->pid_el, KP_EL, KI_EL, KD_EL, MAX_VELOCITY_EL, -MAX_VELOCITY_EL);
    
    memset(&sys->sensor_data, 0, sizeof(sensor_data_t));
    
    sys->location.latitude = latitude;
    sys->location.longitude = longitude;
    sys->location.utc_offset = utc_offset;
    
    sys->last_astro_update = 0;
    sys->current_time = 0;
    
    sys->is_tracking = true;
    sys->is_night = false;
}

/**
 * Main control step
 * Call this function at 10 Hz
 */
static void control_step(control_system_t *sys,
                        const sensor_data_t *new_sensor_data,
                        float *pwm_az_out,
                        float *pwm_el_out) {
    
    float dt = 1.0f / CONTROL_FREQUENCY;
    
    // 1. Update sensor data with filtering
    apply_sensor_filter(&sys->sensor_data, new_sensor_data, 0.3f);
    
    // 2. Calculate sensor-based directional errors
    float delta_az = calculate_azimuth_error(&sys->sensor_data);
    float delta_el = calculate_elevation_error(&sys->sensor_data);
    
    // 3. Update astronomical position (every 60 seconds)
    if (sys->current_time - sys->last_astro_update >= ASTRO_UPDATE_MS) {
        solar_position_t solar_pos;
        calculate_solar_position_complete(&sys->location, &sys->datetime, &solar_pos);
        
        sys->theta_az_astro = solar_pos.azimuth;
        sys->theta_el_astro = solar_pos.altitude;
        
        // Check if night time (altitude < -5 degrees)
        sys->is_night = (solar_pos.altitude < -5.0f);
        
        sys->last_astro_update = sys->current_time;
    }
    
    // 4. Hybrid target calculation (astronomy + sensor feedback)
    if (!sys->is_night && sys->is_tracking) {
        sys->theta_az_target = sys->theta_az_astro + K_AZ_FEEDBACK * delta_az;
        sys->theta_el_target = sys->theta_el_astro + K_EL_FEEDBACK * delta_el;
    } else {
        // Nighttime: park at safe position
        sys->theta_az_target = 180.0f;
        sys->theta_el_target = 0.0f;
    }
    
    // 5. Constrain target positions
    sys->theta_el_target = saturate(sys->theta_el_target, 0.0f, 90.0f);
    
    // 6. Calculate control errors
    float e_az = sys->theta_az_target - sys->theta_az_current;
    float e_el = sys->theta_el_target - sys->theta_el_current;
    
    // Handle azimuth wraparound
    e_az = normalize_angle(e_az);
    
    // 7. Execute PID control
    float u_az = pid_update(&sys->pid_az, e_az, dt);
    float u_el = pid_update(&sys->pid_el, e_el, dt);
    
    // 8. Saturate velocities
    float v_az = saturate(u_az, -MAX_VELOCITY_AZ, MAX_VELOCITY_AZ);
    float v_el = saturate(u_el, -MAX_VELOCITY_EL, MAX_VELOCITY_EL);
    
    // 9. Convert to PWM
    *pwm_az_out = (float)velocity_to_pwm(v_az, MAX_VELOCITY_AZ);
    *pwm_el_out = (float)velocity_to_pwm(v_el, MAX_VELOCITY_EL);
    
    // 10. Update current position (in real implementation, read from encoders)
    // This is a simplified simulation; actual implementation should read from encoders
    sys->theta_az_current += v_az * dt;
    sys->theta_el_current += v_el * dt;
    
    // Normalize azimuth
    if (sys->theta_az_current < 0.0f) sys->theta_az_current += 360.0f;
    if (sys->theta_az_current >= 360.0f) sys->theta_az_current -= 360.0f;
    
    // Constrain elevation
    sys->theta_el_current = saturate(sys->theta_el_current, 0.0f, 90.0f);
}

/* ============================================================================
   DATA LOGGING
   ============================================================================ */

typedef struct {
    uint32_t timestamp_ms;
    float az_cmd;
    float el_cmd;
    float az_actual;
    float el_actual;
    float I_left;
    float I_right;
    float I_upper;
    float I_lower;
    float motor_current_az;
    float motor_current_el;
} log_entry_t;

/**
 * Create log entry from system state
 */
static void create_log_entry(const control_system_t *sys,
                            float motor_current_az,
                            float motor_current_el,
                            log_entry_t *entry) {
    entry->timestamp_ms = sys->current_time;
    entry->az_cmd = sys->theta_az_target;
    entry->el_cmd = sys->theta_el_target;
    entry->az_actual = sys->theta_az_current;
    entry->el_actual = sys->theta_el_current;
    entry->I_left = sys->sensor_data.I_left;
    entry->I_right = sys->sensor_data.I_right;
    entry->I_upper = sys->sensor_data.I_upper;
    entry->I_lower = sys->sensor_data.I_lower;
    entry->motor_current_az = motor_current_az;
    entry->motor_current_el = motor_current_el;
}

/* ============================================================================
   EXAMPLE USAGE
   ============================================================================ */

/*
// Example: Initialize and run control system
int main(void) {
    // Initialize system
    control_system_t sys;
    control_system_init(&sys, 40.0f, -74.0f, -5);  // New York coordinates
    
    // Set current date/time
    sys.datetime.year = 2025;
    sys.datetime.month = 6;
    sys.datetime.day = 21;
    sys.datetime.hour = 12;
    sys.datetime.minute = 0;
    sys.datetime.second = 0;
    
    // Main control loop
    for (int i = 0; i < 36000; i++) {  // 100 hours at 10 Hz
        // Simulate sensor data (in real system, read from ADC)
        sensor_data_t sensor_data;
        sensor_data.I_left = 100.0f;
        sensor_data.I_right = 105.0f;
        sensor_data.I_upper = 102.0f;
        sensor_data.I_lower = 103.0f;
        sensor_data.timestamp = sys.current_time;
        
        // Execute control step
        float pwm_az, pwm_el;
        control_step(&sys, &sensor_data, &pwm_az, &pwm_el);
        
        // Update time
        sys.current_time += 100;  // 100 ms per step
        
        // Update datetime
        uint32_t total_seconds = sys.current_time / 1000;
        sys.datetime.second = total_seconds % 60;
        sys.datetime.minute = (total_seconds / 60) % 60;
        sys.datetime.hour = (total_seconds / 3600) % 24;
        
        // Log data periodically
        if (i % 10 == 0) {
            log_entry_t log;
            create_log_entry(&sys, 0.5f, 0.3f, &log);
            // Write log to file or transmit
        }
    }
    
    return 0;
}
*/

#endif // PHOTOTACTIC_SYSTEM_IMPLEMENTATION_C

