#ifdef _WIN32
#include "d3d11_renderer.h"
#include <iostream>
#include <memory>
#include <sentry.h>
#include <string>

class SentryDirectXDemo {
private:
  bool sentry_initialized;
  std::unique_ptr<D3D11Renderer> renderer;

public:
  SentryDirectXDemo() : sentry_initialized(false) {
    initialize_sentry();
    renderer = std::make_unique<D3D11Renderer>();
  }

  ~SentryDirectXDemo() { cleanup_sentry(); }

  void initialize_sentry() {
    sentry_options_t *options = sentry_options_new();
    sentry_options_set_dsn(options, getenv("SENTRY_DSN"));
    sentry_options_set_database_path(options, ".sentry-native");
    sentry_options_set_release(options, "sentry-directx-demo@1.0.0");
    sentry_options_set_debug(options, 1);

    if (sentry_init(options) == 0) {
      sentry_initialized = true;
      std::cout << "Sentry initialized successfully\n";
    } else {
      std::cerr << "Failed to initialize Sentry\n";
    }
  }

  void report_directx_error(const DirectXError &error,
                            const std::string &crash_type) {
    if (!sentry_initialized)
      return;

    sentry_value_t event = sentry_value_new_event();
    sentry_value_set_by_key(
        event, "message",
        sentry_value_new_string(
            ("DirectX Rendering Crash: " + crash_type).c_str()));
    sentry_value_set_by_key(event, "level", sentry_value_new_string("fatal"));

    sentry_value_t extra = sentry_value_new_object();
    sentry_value_set_by_key(extra, "directx_error_code",
                            sentry_value_new_int32(error.code));
    sentry_value_set_by_key(extra, "directx_error_message",
                            sentry_value_new_string(error.message));
    sentry_value_set_by_key(extra, "directx_error_type",
                            sentry_value_new_string(error.type));
    sentry_value_set_by_key(extra, "crash_scenario",
                            sentry_value_new_string(crash_type.c_str()));
    sentry_value_set_by_key(event, "extra", extra);

    sentry_value_t tags = sentry_value_new_object();
    sentry_value_set_by_key(tags, "component",
                            sentry_value_new_string("directx_renderer"));
    sentry_value_set_by_key(tags, "crash_type",
                            sentry_value_new_string(crash_type.c_str()));
    sentry_value_set_by_key(tags, "platform",
                            sentry_value_new_string("windows"));
    sentry_value_set_by_key(event, "tags", tags);

    sentry_capture_event(event);
    std::cout << "Reported DirectX error to Sentry: " << crash_type << "\n";
  }

  void test_invalid_buffer_access() {
    std::cout << "\n--- Testing Invalid Buffer Access ---\n";
    DirectXError error = renderer->test_invalid_buffer_access();
    if (error.code != 0) {
      report_directx_error(error, "invalid_buffer_access");
    }
  }

  void test_invalid_shader_resource() {
    std::cout << "\n--- Testing Invalid Shader Resource ---\n";
    DirectXError error = renderer->test_invalid_shader_resource();
    if (error.code != 0) {
      report_directx_error(error, "invalid_shader_resource");
    }
  }

  void test_device_removed_simulation() {
    std::cout << "\n--- Testing Device Removed Simulation ---\n";
    DirectXError error = renderer->test_device_removed_simulation();
    if (error.code != 0) {
      report_directx_error(error, "device_removed_simulation");
    }
  }

  void test_invalid_render_target() {
    std::cout << "\n--- Testing Invalid Render Target ---\n";
    DirectXError error = renderer->test_invalid_render_target();
    if (error.code != 0) {
      report_directx_error(error, "invalid_render_target");
    }
  }

  void test_out_of_bounds_vertex_buffer() {
    std::cout << "\n--- Testing Out of Bounds Vertex Buffer ---\n";
    DirectXError error = renderer->test_out_of_bounds_vertex_buffer();
    if (error.code != 0) {
      report_directx_error(error, "out_of_bounds_vertex_buffer");
    }
  }

  bool initialize_renderer() {
    if (!renderer->initialize()) {
      std::cerr << "Failed to initialize DirectX renderer\n";
      return false;
    }
    return true;
  }

  void run_all_tests() {
    std::cout
        << "Starting DirectX crash demonstration with Sentry reporting...\n";

    if (!initialize_renderer()) {
      std::cerr << "Cannot run tests without initialized renderer\n";
      return;
    }

    test_invalid_buffer_access();
    test_invalid_shader_resource();
    test_device_removed_simulation();
    test_invalid_render_target();
    test_out_of_bounds_vertex_buffer();

    std::cout << "\nAll DirectX tests completed. Check Sentry dashboard for "
                 "reported errors.\n";
  }

private:
  void cleanup_sentry() {
    if (sentry_initialized) {
      sentry_close();
    }
  }
};

int main(int argc, char *argv[]) {
  try {
    if (!getenv("SENTRY_DSN")) {
      std::cout << "Warning: SENTRY_DSN environment variable not set. Errors "
                   "will not be reported to Sentry.\n";
      std::cout << "Set SENTRY_DSN to your Sentry project DSN to enable error "
                   "reporting.\n\n";
    }

    SentryDirectXDemo demo;

    if (argc > 1) {
      std::string test_type = argv[1];
      if (!demo.initialize_renderer()) {
        std::cerr << "Failed to initialize DirectX renderer\n";
        return 1;
      }

      if (test_type == "invalid_buffer_access") {
        demo.test_invalid_buffer_access();
      } else if (test_type == "invalid_shader_resource") {
        demo.test_invalid_shader_resource();
      } else if (test_type == "device_removed_simulation") {
        demo.test_device_removed_simulation();
      } else if (test_type == "invalid_render_target") {
        demo.test_invalid_render_target();
      } else if (test_type == "out_of_bounds_vertex_buffer") {
        demo.test_out_of_bounds_vertex_buffer();
      } else {
        std::cout << "Unknown test type: " << test_type << std::endl;
        std::cout << "Available tests: invalid_buffer_access, "
                     "invalid_shader_resource, device_removed_simulation, "
                     "invalid_render_target, out_of_bounds_vertex_buffer"
                  << std::endl;
        return 1;
      }
    } else {
      demo.run_all_tests();
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#else
#include <iostream>

int main() {
  std::cout << "DirectX demo is only available on Windows" << std::endl;
  return 0;
}

#endif // _WIN32