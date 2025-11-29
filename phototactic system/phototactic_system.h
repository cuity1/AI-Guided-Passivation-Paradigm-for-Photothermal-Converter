/**
 * Biomimetic Dual-Axis Phototactic System - Header File
 * 
 * This header file defines the public interface for the solar tracking system.
 * It includes function declarations, data structures, and configuration constants.
 */

#ifndef PHOTOTACTIC_SYSTEM_H
#define PHOTOTACTIC_SYSTEM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
   CONFIGURATION CONSTANTS
   ============================================================================ */

// System timing
#define PHOTOTACTIC_CONTROL_FREQ_HZ    10      // Control loop frequency (Hz)
#define PHOTOTACTIC_SENSOR_FREQ_HZ     10      // Sensor sampling frequency (Hz)
#define PHOTOTACTIC_ASTRO_UPDATE_HZ    1       // Astronomical update frequency (Hz)

// Motor specifications
#define PHOTOTACTIC_MAX_VEL_AZ_DPS     18.0f   // Max azimuth velocity (deg/s)
#define PHOTOTACTIC_MAX_VEL_EL_DPS     12.0f   // Max elevation velocity (deg/s)
#define PHOTOTACTIC_MAX_ACCEL_DPS2     5.0f    // Max acceleration (deg/s²)

// Sensor specifications
#define PHOTOTACTIC_SENSOR_COUNT       4       // Number of sensors in array
#define PHOTOTACTIC_SENSOR_FILTER_HZ   5.0f    // Low-pass filter cutoff (Hz)

// Control gains
#define PHOTOTACTIC_KP_AZ              0.8f
#define PHOTOTACTIC_KI_AZ              0.05f
#define PHOTOTACTIC_KD_AZ              0.2f
#define PHOTOTACTIC_KP_EL              0.8f
#define PHOTOTACTIC_KI_EL              0.05f
#define PHOTOTACTIC_KD_EL              0.2f

// Feedback gains
#define PHOTOTACTIC_K_AZ_FEEDBACK      0.6f
#define PHOTOTACTIC_K_EL_FEEDBACK      0.6f

// Astronomical algorithm accuracy
#define PHOTOTACTIC_DECLINATION_ACCURACY_DEG  0.1f
#define PHOTOTACTIC_POSITION_ACCURACY_DEG     0.3f

// System constraints
#define PHOTOTACTIC_AZ_MIN_DEG         0.0f
#define PHOTOTACTIC_AZ_MAX_DEG         360.0f
#define PHOTOTACTIC_EL_MIN_DEG         0.0f
#define PHOTOTACTIC_EL_MAX_DEG         90.0f
#define PHOTOTACTIC_NIGHT_THRESHOLD_DEG (-5.0f)

/* ============================================================================
   DATA STRUCTURES
   ============================================================================ */

/**
 * Sensor array data structure
 */
typedef struct {
    float I_left;           // Irradiance at left sensor (W/m²)
    float I_right;          // Irradiance at right sensor (W/m²)
    float I_upper;          // Irradiance at upper sensor (W/m²)
    float I_lower;          // Irradiance at lower sensor (W/m²)
    uint32_t timestamp_ms;  // Measurement timestamp (milliseconds)
} phototactic_sensor_data_t;

/**
 * Solar position data structure
 */
typedef struct {
    float azimuth;          // Solar azimuth angle (degrees, 0=N, 90=E)
    float altitude;         // Solar altitude angle (degrees, 0=horizon, 90=zenith)
    float declination;      // Solar declination (degrees)
    float hour_angle;       // Hour angle (degrees)
} phototactic_solar_position_t;

/**
 * Observer location data structure
 */
typedef struct {
    float latitude;         // Observer latitude (degrees, -90 to +90)
    float longitude;        // Observer longitude (degrees, -180 to +180)
    int32_t utc_offset_hours; // UTC offset (hours)
} phototactic_location_t;

/**
 * Date and time data structure
 */
typedef struct {
    uint16_t year;          // Year (e.g., 2025)
    uint8_t month;          // Month (1-12)
    uint8_t day;            // Day (1-31)
    uint8_t hour;           // Hour (0-23)
    uint8_t minute;         // Minute (0-59)
    uint8_t second;         // Second (0-59)
} phototactic_datetime_t;

/**
 * System state data structure
 */
typedef struct {
    float azimuth_angle;    // Current azimuth angle (degrees)
    float elevation_angle;  // Current elevation angle (degrees)
    float azimuth_velocity; // Current azimuth velocity (deg/s)
    float elevation_velocity; // Current elevation velocity (deg/s)
    bool is_tracking;       // Tracking enabled flag
    bool is_night;          // Night time flag
    float tracking_error_az; // Azimuth tracking error (degrees)
    float tracking_error_el; // Elevation tracking error (degrees)
} phototactic_system_state_t;

/**
 * Control output data structure
 */
typedef struct {
    uint8_t pwm_azimuth;    // Azimuth motor PWM (0-255)
    uint8_t pwm_elevation;  // Elevation motor PWM (0-255)
    float velocity_az;      // Azimuth velocity command (deg/s)
    float velocity_el;      // Elevation velocity command (deg/s)
} phototactic_control_output_t;

/**
 * System configuration data structure
 */
typedef struct {
    phototactic_location_t location;
    float pid_kp_az;
    float pid_ki_az;
    float pid_kd_az;
    float pid_kp_el;
    float pid_ki_el;
    float pid_kd_el;
    float feedback_gain_az;
    float feedback_gain_el;
    float max_velocity_az;
    float max_velocity_el;
    float max_acceleration;
} phototactic_config_t;

/* ============================================================================
   FUNCTION DECLARATIONS
   ============================================================================ */

/**
 * Initialize the phototactic control system
 * 
 * @param config Pointer to system configuration structure
 * @return 0 on success, non-zero on error
 */
int phototactic_init(const phototactic_config_t *config);

/**
 * Deinitialize the phototactic control system
 * 
 * @return 0 on success, non-zero on error
 */
int phototactic_deinit(void);

/**
 * Execute one control step (should be called at CONTROL_FREQUENCY)
 * 
 * @param sensor_data Pointer to current sensor data
 * @param datetime Pointer to current date/time
 * @param output Pointer to control output structure (filled by function)
 * @return 0 on success, non-zero on error
 */
int phototactic_control_step(const phototactic_sensor_data_t *sensor_data,
                             const phototactic_datetime_t *datetime,
                             phototactic_control_output_t *output);

/**
 * Get current system state
 * 
 * @param state Pointer to system state structure (filled by function)
 * @return 0 on success, non-zero on error
 */
int phototactic_get_state(phototactic_system_state_t *state);

/**
 * Set system configuration parameters
 * 
 * @param config Pointer to new configuration
 * @return 0 on success, non-zero on error
 */
int phototactic_set_config(const phototactic_config_t *config);

/**
 * Enable/disable tracking
 * 
 * @param enable true to enable tracking, false to disable
 * @return 0 on success, non-zero on error
 */
int phototactic_set_tracking_enabled(bool enable);

/**
 * Reset control system state
 * 
 * @return 0 on success, non-zero on error
 */
int phototactic_reset(void);

/**
 * Calculate solar position for given location and time
 * 
 * @param location Pointer to observer location
 * @param datetime Pointer to date/time
 * @param position Pointer to solar position structure (filled by function)
 * @return 0 on success, non-zero on error
 */
int phototactic_calculate_solar_position(const phototactic_location_t *location,
                                         const phototactic_datetime_t *datetime,
                                         phototactic_solar_position_t *position);

/**
 * Calculate directional error from sensor array
 * 
 * @param sensor_data Pointer to sensor data
 * @param error_az Pointer to azimuth error (filled by function)
 * @param error_el Pointer to elevation error (filled by function)
 * @return 0 on success, non-zero on error
 */
int phototactic_calculate_sensor_error(const phototactic_sensor_data_t *sensor_data,
                                       float *error_az,
                                       float *error_el);

/**
 * Perform system calibration
 * 
 * @return 0 on success, non-zero on error
 */
int phototactic_calibrate(void);

/**
 * Get system diagnostics information
 * 
 * @param diagnostics Pointer to diagnostics string buffer
 * @param buffer_size Size of diagnostics buffer
 * @return Number of bytes written, negative on error
 */
int phototactic_get_diagnostics(char *diagnostics, size_t buffer_size);

/* ============================================================================
   UTILITY FUNCTIONS
   ============================================================================ */

/**
 * Convert degrees to radians
 */
float phototactic_deg_to_rad(float degrees);

/**
 * Convert radians to degrees
 */
float phototactic_rad_to_deg(float radians);

/**
 * Normalize angle to [-180, 180] range
 */
float phototactic_normalize_angle(float angle);

/**
 * Saturate value between min and max
 */
float phototactic_saturate(float value, float min, float max);

/* ============================================================================
   ERROR CODES
   ============================================================================ */

#define PHOTOTACTIC_SUCCESS             0
#define PHOTOTACTIC_ERROR_INVALID_PARAM -1
#define PHOTOTACTIC_ERROR_NOT_INITIALIZED -2
#define PHOTOTACTIC_ERROR_SENSOR_FAULT  -3
#define PHOTOTACTIC_ERROR_MOTOR_FAULT   -4
#define PHOTOTACTIC_ERROR_ENCODER_FAULT -5
#define PHOTOTACTIC_ERROR_MEMORY        -6
#define PHOTOTACTIC_ERROR_TIMEOUT       -7

#ifdef __cplusplus
}
#endif

#endif // PHOTOTACTIC_SYSTEM_H


