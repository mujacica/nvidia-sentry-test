#pragma once

#ifdef _WIN32
#include <DirectXMath.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <string>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

struct DirectXError {
  int code;
  const char *message;
  const char *type;
};

class D3D11Renderer {
public:
  D3D11Renderer();
  ~D3D11Renderer();

  bool initialize(HWND hwnd = nullptr);
  void cleanup();

  DirectXError test_invalid_buffer_access();
  DirectXError test_invalid_shader_resource();
  DirectXError test_device_removed_simulation();
  DirectXError test_invalid_render_target();
  DirectXError test_out_of_bounds_vertex_buffer();

  // NVIDIA Driver crash scenarios - WARNING: These may crash the driver or
  // system
  DirectXError test_nvidia_driver_memory_exhaustion();
  DirectXError test_nvidia_driver_invalid_resource_binding();
  DirectXError test_nvidia_driver_command_list_corruption();
  DirectXError test_nvidia_driver_excessive_draw_calls();
  DirectXError test_nvidia_driver_concurrent_context_destruction();

private:
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  ComPtr<IDXGISwapChain> swapChain;
  ComPtr<ID3D11RenderTargetView> renderTargetView;
  ComPtr<ID3D11Buffer> vertexBuffer;
  ComPtr<ID3D11Buffer> indexBuffer;
  ComPtr<ID3D11VertexShader> vertexShader;
  ComPtr<ID3D11PixelShader> pixelShader;
  ComPtr<ID3D11InputLayout> inputLayout;

  bool createDevice();
  bool createSwapChain(HWND hwnd);
  bool createRenderTargetView();
  bool createBuffers();
  bool createShaders();

  static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam,
                                     LPARAM lParam);
  HWND createDummyWindow();

  HWND dummyWindow = nullptr;
  static const char *vertexShaderSource;
  static const char *pixelShaderSource;
};

#endif // _WIN32