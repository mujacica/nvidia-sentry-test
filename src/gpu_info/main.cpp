#include <chrono>
#include <iostream>
#include <sentry.h>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi.h>
#include <windows.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#elif __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <Metal/Metal.h>
#else

// Linux - try to include common GPU headers
#ifdef __has_include
#if __has_include(<GL/gl.h>)
#include <GL/gl.h>
#endif
#endif
#include <cstring>
#include <sys/utsname.h>
#endif

void print_gpu_info() {
  std::cout << "=== GPU Information Collection Demo ===" << std::endl;
  std::cout << "Platform: ";

#ifdef _WIN32
  std::cout << "Windows" << std::endl;

  // DirectX/DXGI GPU enumeration
  IDXGIFactory *factory = nullptr;
  HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void **)&factory);

  if (SUCCEEDED(hr)) {
    std::cout << "Available GPU devices (via DXGI):" << std::endl;

    IDXGIAdapter *adapter = nullptr;
    UINT i = 0;

    while (factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND) {
      DXGI_ADAPTER_DESC desc;
      adapter->GetDesc(&desc);

      std::wcout << L"  [" << i << L"] " << desc.Description << L" (VRAM: "
                 << (desc.DedicatedVideoMemory / 1024 / 1024) << L" MB)"
                 << std::endl;

      adapter->Release();
      ++i;
    }

    factory->Release();
  } else {
    std::cout << "Could not initialize DXGI for GPU enumeration" << std::endl;
  }

#elif __APPLE__
  std::cout << "macOS" << std::endl;

  // Metal GPU enumeration
  @autoreleasepool {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    std::cout << "Available GPU devices (via Metal):" << std::endl;

    for (NSUInteger i = 0; i < [devices count]; i++) {
      id<MTLDevice> device = [devices objectAtIndex:i];
      NSString *name = [device name];
      const char *cName = [name UTF8String];

      std::cout << "  [" << i << "] " << cName;

      if (@available(macOS 10.15, *)) {
        std::cout << " (Memory: "
                  << ([device recommendedMaxWorkingSetSize] / 1024 / 1024)
                  << " MB)";
      }
      std::cout << std::endl;
    }

    [devices release];
  }

#else
  std::cout << "Linux" << std::endl;

  // Basic system info
  struct utsname sys_info;
  if (uname(&sys_info) == 0) {
    std::cout << "System: " << sys_info.sysname << " " << sys_info.release
              << std::endl;
  }

#ifdef HAS_OPENGL
  std::cout << "OpenGL headers available - GPU detection possible" << std::endl;
#else
  std::cout << "Limited GPU detection available on this system" << std::endl;
#endif

  // Try to read GPU info from /proc or /sys
  std::cout << "Attempting to detect GPU from system files..." << std::endl;
  // Note: This is basic detection - real applications would use proper GPU APIs

#endif

  std::cout << std::endl;
}

void send_test_event(const std::string &event_type) {
  std::cout << "Sending Sentry event: " << event_type << std::endl;

  sentry_value_t event = sentry_value_new_event();
  sentry_value_set_by_key(
      event, "message",
      sentry_value_new_string(("GPU Info Test - " + event_type).c_str()));
  sentry_value_set_by_key(event, "level", sentry_value_new_string("info"));

  // Add custom tags
  sentry_value_t tags = sentry_value_new_object();
  sentry_value_set_by_key(tags, "component",
                          sentry_value_new_string("gpu_info"));
  sentry_value_set_by_key(tags, "test_type",
                          sentry_value_new_string(event_type.c_str()));

#ifdef _WIN32
  sentry_value_set_by_key(tags, "platform", sentry_value_new_string("windows"));
#elif __APPLE__
  sentry_value_set_by_key(tags, "platform", sentry_value_new_string("macos"));
#else
  sentry_value_set_by_key(tags, "platform", sentry_value_new_string("linux"));
#endif

  sentry_value_set_by_key(event, "tags", tags);

  // Add extra context
  sentry_value_t extra = sentry_value_new_object();
  sentry_value_set_by_key(
      extra, "demo_purpose",
      sentry_value_new_string("Test GPU info collection in Sentry events"));
  sentry_value_set_by_key(
      extra, "timestamp",
      sentry_value_new_string(
          std::to_string(
              std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count())
              .c_str()));
  sentry_value_set_by_key(event, "extra", extra);

  // Send the event
  sentry_uuid_t event_id = sentry_capture_event(event);

  char event_id_str[37];
  sentry_uuid_as_string(&event_id, event_id_str);
  std::cout << "Event sent with ID: " << event_id_str << std::endl;
}

int main(int argc, char *argv[]) {
  std::cout << "Sentry GPU Info Collection Demo" << std::endl;
  std::cout << "===============================" << std::endl << std::endl;

  // Initialize Sentry
  sentry_options_t *options = sentry_options_new();

  const char *dsn = std::getenv("SENTRY_DSN");
  if (!dsn || strlen(dsn) == 0) {
    std::cerr << "Error: SENTRY_DSN environment variable not set!" << std::endl;
    std::cerr << "Usage: export SENTRY_DSN='your-sentry-dsn-here'" << std::endl;
    std::cerr << "       " << argv[0] << " [test_type]" << std::endl;
    return 1;
  }

  sentry_options_set_dsn(options, dsn);
  sentry_options_set_release(options, "gpu-info-demo@1.0.0");
  sentry_options_set_environment(options, "development");

  // Enable debug mode for verbose output
  sentry_options_set_debug(options, 1);

  int init_result = sentry_init(options);
  if (init_result != 0) {
    std::cerr << "Failed to initialize Sentry!" << std::endl;
    return 1;
  }

  std::cout
      << "Sentry initialized successfully with GPU info collection enabled"
      << std::endl;
  std::cout << "DSN: " << dsn << std::endl << std::endl;

  // Print GPU information
  print_gpu_info();

  // Determine test type
  std::string test_type = "basic";
  if (argc > 1) {
    test_type = argv[1];
  }

  std::cout << "Running test type: " << test_type << std::endl << std::endl;

  if (test_type == "basic" || test_type == "all") {
    send_test_event("basic_info");
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  if (test_type == "warning" || test_type == "all") {
    // Send a warning event
    sentry_value_t event = sentry_value_new_event();
    sentry_value_set_by_key(
        event, "message",
        sentry_value_new_string("GPU Info Test - Warning Level"));
    sentry_value_set_by_key(event, "level", sentry_value_new_string("warning"));

    sentry_value_t tags = sentry_value_new_object();
    sentry_value_set_by_key(tags, "component",
                            sentry_value_new_string("gpu_info"));
    sentry_value_set_by_key(tags, "test_type",
                            sentry_value_new_string("warning"));
    sentry_value_set_by_key(event, "tags", tags);

    sentry_uuid_t event_id = sentry_capture_event(event);
    char event_id_str[37];
    sentry_uuid_as_string(&event_id, event_id_str);
    std::cout << "Warning event sent with ID: " << event_id_str << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  if (test_type == "error" || test_type == "all") {
    // Send an error event
    sentry_value_t event = sentry_value_new_event();
    sentry_value_set_by_key(
        event, "message",
        sentry_value_new_string("GPU Info Test - Error Level"));
    sentry_value_set_by_key(event, "level", sentry_value_new_string("error"));

    sentry_value_t tags = sentry_value_new_object();
    sentry_value_set_by_key(tags, "component",
                            sentry_value_new_string("gpu_info"));
    sentry_value_set_by_key(tags, "test_type",
                            sentry_value_new_string("error"));
    sentry_value_set_by_key(event, "tags", tags);

    sentry_uuid_t event_id = sentry_capture_event(event);
    char event_id_str[37];
    sentry_uuid_as_string(&event_id, event_id_str);
    std::cout << "Error event sent with ID: " << event_id_str << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  if (test_type == "exception" || test_type == "all") {
    // Trigger an actual exception to test crash reporting
    std::cout << "Triggering exception to test GPU info in crash reports..."
              << std::endl;
    try {
      throw std::runtime_error(
          "Intentional exception to test GPU info collection");
    } catch (const std::exception &e) {
      sentry_value_t event = sentry_value_new_event();
      sentry_value_set_by_key(event, "message",
                              sentry_value_new_string(e.what()));
      sentry_value_set_by_key(event, "level", sentry_value_new_string("error"));

      sentry_value_t tags = sentry_value_new_object();
      sentry_value_set_by_key(tags, "component",
                              sentry_value_new_string("gpu_info"));
      sentry_value_set_by_key(tags, "test_type",
                              sentry_value_new_string("exception"));
      sentry_value_set_by_key(event, "tags", tags);

      sentry_uuid_t event_id = sentry_capture_event(event);
      char event_id_str[37];
      sentry_uuid_as_string(&event_id, event_id_str);
      std::cout << "Exception event sent with ID: " << event_id_str
                << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  if (test_type == "crash" || test_type == "all") {
    // Trigger an unhandled exception that crashpad should catch
    std::cout << "Triggering unhandled exception for crashpad to catch..."
              << std::endl;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(500)); // Give time for message to print

    // This will cause an unhandled exception and trigger crashpad
    volatile int *null_ptr = nullptr;
    *null_ptr = 42; // Deliberate segmentation fault for crash reporting
  }

  std::cout << std::endl
            << "All events sent! Check your Sentry dashboard to verify GPU "
               "info is included."
            << std::endl;
  std::cout << "GPU information should appear in the 'Device' or 'Contexts' "
               "section of each event."
            << std::endl;

  // Flush events and shutdown
  sentry_flush(5000); // Wait up to 5 seconds for events to be sent
  sentry_close();

  std::cout << std::endl << "Demo completed successfully!" << std::endl;

  return 0;
}