#ifdef _WIN32
#include "d3d11_renderer.h"
#include <iostream>
#include <vector>

const char *D3D11Renderer::vertexShaderSource = R"(
struct VertexInput {
    float3 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

VertexOutput main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 1.0f);
    output.color = input.color;
    return output;
}
)";

const char *D3D11Renderer::pixelShaderSource = R"(
struct PixelInput {
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

float4 main(PixelInput input) : SV_TARGET {
    return input.color;
}
)";

D3D11Renderer::D3D11Renderer() {}

D3D11Renderer::~D3D11Renderer() { cleanup(); }

bool D3D11Renderer::initialize(HWND hwnd) {
  if (hwnd == nullptr) {
    hwnd = createDummyWindow();
    if (hwnd == nullptr) {
      std::cerr << "Failed to create dummy window" << std::endl;
      return false;
    }
    dummyWindow = hwnd;
  }

  if (!createDevice()) {
    std::cerr << "Failed to create D3D11 device" << std::endl;
    return false;
  }

  if (!createSwapChain(hwnd)) {
    std::cerr << "Failed to create swap chain" << std::endl;
    return false;
  }

  if (!createRenderTargetView()) {
    std::cerr << "Failed to create render target view" << std::endl;
    return false;
  }

  if (!createBuffers()) {
    std::cerr << "Failed to create buffers" << std::endl;
    return false;
  }

  if (!createShaders()) {
    std::cerr << "Failed to create shaders" << std::endl;
    return false;
  }

  std::cout << "DirectX 11 renderer initialized successfully" << std::endl;
  return true;
}

void D3D11Renderer::cleanup() {
  if (context) {
    context->ClearState();
  }

  if (dummyWindow) {
    DestroyWindow(dummyWindow);
    dummyWindow = nullptr;
  }
}

DirectXError D3D11Renderer::test_invalid_buffer_access() {
  ID3D11Buffer *invalidBuffer = nullptr;
  UINT stride = sizeof(float) * 7; // position + color
  UINT offset = 1000000;           // Invalid large offset

  context->IASetVertexBuffers(0, 1, &invalidBuffer, &stride, &offset);

  D3D11_MAPPED_SUBRESOURCE mappedResource;
  HRESULT hr = context->Map(invalidBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0,
                            &mappedResource);

  if (FAILED(hr)) {
    return {static_cast<int>(hr), "Invalid buffer access error",
            "D3D11_INVALID_BUFFER_ACCESS"};
  }

  context->Unmap(invalidBuffer, 0);
  return {0, "Success", "NONE"};
}

DirectXError D3D11Renderer::test_invalid_shader_resource() {
  ID3D11ShaderResourceView *invalidSRV = nullptr;

  context->PSSetShaderResources(0, 1, &invalidSRV);

  context->Draw(3, 0);

  HRESULT hr = device->GetDeviceRemovedReason();
  if (FAILED(hr)) {
    return {static_cast<int>(hr), "Invalid shader resource error",
            "D3D11_INVALID_SHADER_RESOURCE"};
  }

  return {0, "Success", "NONE"};
}

DirectXError D3D11Renderer::test_device_removed_simulation() {
  std::vector<ComPtr<ID3D11Buffer>> buffers;

  // Try to allocate excessive memory to trigger device removal
  for (int i = 0; i < 10000; i++) {
    ComPtr<ID3D11Buffer> buffer;

    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = 1024 * 1024 * 100; // 100MB per buffer
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    HRESULT hr = device->CreateBuffer(&desc, nullptr, &buffer);
    if (FAILED(hr)) {
      return {static_cast<int>(hr), "Device removed simulation error",
              "D3D11_DEVICE_REMOVED_SIMULATION"};
    }

    buffers.push_back(buffer);
  }

  return {0, "Success", "NONE"};
}

DirectXError D3D11Renderer::test_invalid_render_target() {
  ID3D11RenderTargetView *invalidRTV = nullptr;

  context->OMSetRenderTargets(1, &invalidRTV, nullptr);

  float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  context->ClearRenderTargetView(invalidRTV, clearColor);

  HRESULT hr = device->GetDeviceRemovedReason();
  if (FAILED(hr)) {
    return {static_cast<int>(hr), "Invalid render target error",
            "D3D11_INVALID_RENDER_TARGET"};
  }

  return {0, "Success", "NONE"};
}

DirectXError D3D11Renderer::test_out_of_bounds_vertex_buffer() {
  if (!vertexBuffer) {
    return {-1, "Vertex buffer not created", "D3D11_NO_VERTEX_BUFFER"};
  }

  UINT stride = sizeof(float) * 7;
  UINT offset = 0;

  context->IASetVertexBuffers(0, 1, vertexBuffer.GetAddressOf(), &stride,
                              &offset);

  // Draw more vertices than actually exist in the buffer
  context->Draw(1000000, 0);

  HRESULT hr = device->GetDeviceRemovedReason();
  if (FAILED(hr)) {
    return {static_cast<int>(hr), "Out of bounds vertex buffer error",
            "D3D11_OUT_OF_BOUNDS_VERTEX_BUFFER"};
  }

  return {0, "Success", "NONE"};
}

bool D3D11Renderer::createDevice() {
  D3D_FEATURE_LEVEL featureLevels[] = {
      D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1,
      D3D_FEATURE_LEVEL_10_0};

  UINT createDeviceFlags = 0;
#ifdef _DEBUG
  createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

  D3D_FEATURE_LEVEL featureLevel;
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                 createDeviceFlags, featureLevels,
                                 ARRAYSIZE(featureLevels), D3D11_SDK_VERSION,
                                 &device, &featureLevel, &context);

  return SUCCEEDED(hr);
}

bool D3D11Renderer::createSwapChain(HWND hwnd) {
  ComPtr<IDXGIDevice> dxgiDevice;
  HRESULT hr = device.As(&dxgiDevice);
  if (FAILED(hr))
    return false;

  ComPtr<IDXGIAdapter> adapter;
  hr = dxgiDevice->GetAdapter(&adapter);
  if (FAILED(hr))
    return false;

  ComPtr<IDXGIFactory> factory;
  hr = adapter->GetParent(IID_PPV_ARGS(&factory));
  if (FAILED(hr))
    return false;

  DXGI_SWAP_CHAIN_DESC desc = {};
  desc.BufferDesc.Width = 800;
  desc.BufferDesc.Height = 600;
  desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  desc.BufferDesc.RefreshRate.Numerator = 60;
  desc.BufferDesc.RefreshRate.Denominator = 1;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount = 1;
  desc.OutputWindow = hwnd;
  desc.Windowed = TRUE;
  desc.SampleDesc.Count = 1;
  desc.SampleDesc.Quality = 0;

  hr = factory->CreateSwapChain(device.Get(), &desc, &swapChain);
  return SUCCEEDED(hr);
}

bool D3D11Renderer::createRenderTargetView() {
  ComPtr<ID3D11Texture2D> backBuffer;
  HRESULT hr = swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
  if (FAILED(hr))
    return false;

  hr = device->CreateRenderTargetView(backBuffer.Get(), nullptr,
                                      &renderTargetView);
  return SUCCEEDED(hr);
}

bool D3D11Renderer::createBuffers() {
  // Vertex data (position + color)
  float vertices[] = {
      -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // Bottom left, red
      0.5f,  -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // Bottom right, green
      0.0f,  0.5f,  0.0f, 0.0f, 0.0f, 1.0f, 1.0f  // Top center, blue
  };

  D3D11_BUFFER_DESC desc = {};
  desc.ByteWidth = sizeof(vertices);
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

  D3D11_SUBRESOURCE_DATA initData = {};
  initData.pSysMem = vertices;

  HRESULT hr = device->CreateBuffer(&desc, &initData, &vertexBuffer);
  return SUCCEEDED(hr);
}

bool D3D11Renderer::createShaders() {
  ComPtr<ID3DBlob> vsBlob, psBlob, errorBlob;

  // Compile vertex shader
  HRESULT hr = D3DCompile(vertexShaderSource, strlen(vertexShaderSource),
                          nullptr, nullptr, nullptr, "main", "vs_5_0",
                          D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0,
                          &vsBlob, &errorBlob);

  if (FAILED(hr)) {
    if (errorBlob) {
      std::cerr << "Vertex shader compilation error: "
                << (char *)errorBlob->GetBufferPointer() << std::endl;
    }
    return false;
  }

  hr = device->CreateVertexShader(vsBlob->GetBufferPointer(),
                                  vsBlob->GetBufferSize(), nullptr,
                                  &vertexShader);
  if (FAILED(hr))
    return false;

  // Compile pixel shader
  hr = D3DCompile(pixelShaderSource, strlen(pixelShaderSource), nullptr,
                  nullptr, nullptr, "main", "ps_5_0",
                  D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, &psBlob,
                  &errorBlob);

  if (FAILED(hr)) {
    if (errorBlob) {
      std::cerr << "Pixel shader compilation error: "
                << (char *)errorBlob->GetBufferPointer() << std::endl;
    }
    return false;
  }

  hr =
      device->CreatePixelShader(psBlob->GetBufferPointer(),
                                psBlob->GetBufferSize(), nullptr, &pixelShader);
  if (FAILED(hr))
    return false;

  // Create input layout
  D3D11_INPUT_ELEMENT_DESC inputElements[] = {
      {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
       D3D11_INPUT_PER_VERTEX_DATA, 0},
      {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,
       D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0}};

  hr = device->CreateInputLayout(inputElements, ARRAYSIZE(inputElements),
                                 vsBlob->GetBufferPointer(),
                                 vsBlob->GetBufferSize(), &inputLayout);

  return SUCCEEDED(hr);
}

LRESULT CALLBACK D3D11Renderer::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam,
                                           LPARAM lParam) {
  switch (uMsg) {
  case WM_DESTROY:
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

HWND D3D11Renderer::createDummyWindow() {
  WNDCLASS wc = {};
  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(nullptr);
  wc.lpszClassName = L"SentryDirectXDemoWindow";

  if (!RegisterClass(&wc)) {
    return nullptr;
  }

  return CreateWindowEx(0, L"SentryDirectXDemoWindow", L"Sentry DirectX Demo",
                        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800,
                        600, nullptr, nullptr, GetModuleHandle(nullptr),
                        nullptr);
}

// NVIDIA Driver crash scenarios - WARNING: These may crash the driver or system
DirectXError D3D11Renderer::test_nvidia_driver_memory_exhaustion() {
  // Attempt to allocate massive amounts of GPU memory to exhaust NVIDIA driver
  std::vector<ComPtr<ID3D11Buffer>> buffers;
  std::vector<ComPtr<ID3D11Texture2D>> textures;

  // Try to allocate many large buffers
  for (int i = 0; i < 1000; i++) {
    ComPtr<ID3D11Buffer> buffer;

    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = 256 * 1024 * 1024; // 256MB per buffer
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_INDEX_BUFFER;
    desc.CPUAccessFlags = 0;

    HRESULT hr = device->CreateBuffer(&desc, nullptr, &buffer);
    if (FAILED(hr)) {
      return {static_cast<int>(hr), "NVIDIA driver memory exhaustion",
              "D3D11_NVIDIA_DRIVER_MEMORY_EXHAUSTION"};
    }

    buffers.push_back(buffer);

    // Also create large textures
    ComPtr<ID3D11Texture2D> texture;
    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = 4096;
    texDesc.Height = 4096;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    hr = device->CreateTexture2D(&texDesc, nullptr, &texture);
    if (FAILED(hr)) {
      return {static_cast<int>(hr), "NVIDIA driver memory exhaustion",
              "D3D11_NVIDIA_DRIVER_MEMORY_EXHAUSTION"};
    }

    textures.push_back(texture);
  }

  // Resources will be automatically released when ComPtr goes out of scope
  return {0, "Success", "NONE"};
}

DirectXError D3D11Renderer::test_nvidia_driver_invalid_resource_binding() {
  // Create invalid resource bindings to trigger NVIDIA driver issues

  // Create a buffer that we'll immediately destroy
  ComPtr<ID3D11Buffer> buffer;
  D3D11_BUFFER_DESC desc = {};
  desc.ByteWidth = 4096;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

  HRESULT hr = device->CreateBuffer(&desc, nullptr, &buffer);
  if (FAILED(hr)) {
    return {static_cast<int>(hr), "Failed to create buffer",
            "D3D11_NVIDIA_DRIVER_INVALID_RESOURCE_BINDING"};
  }

  // Get the raw pointer before releasing the ComPtr
  ID3D11Buffer *rawBuffer = buffer.Get();
  buffer.Reset(); // Release the ComPtr, making the buffer invalid

  // Try to bind the now-invalid buffer
  UINT stride = sizeof(float) * 7;
  UINT offset = 0;
  context->IASetVertexBuffers(0, 1, &rawBuffer, &stride, &offset);

  // Try to draw with invalid buffer bound
  context->Draw(3, 0);

  // Check for device errors
  HRESULT deviceResult = device->GetDeviceRemovedReason();
  if (FAILED(deviceResult)) {
    return {static_cast<int>(deviceResult), "Invalid resource binding",
            "D3D11_NVIDIA_DRIVER_INVALID_RESOURCE_BINDING"};
  }

  return {0, "Success", "NONE"};
}

DirectXError D3D11Renderer::test_nvidia_driver_command_list_corruption() {
  // Create many simultaneous command contexts to corrupt NVIDIA driver state
  std::vector<ComPtr<ID3D11DeviceContext>> deferredContexts;
  std::vector<ComPtr<ID3D11CommandList>> commandLists;

  // Create many deferred contexts
  for (int i = 0; i < 100; i++) {
    ComPtr<ID3D11DeviceContext> deferredContext;
    HRESULT hr = device->CreateDeferredContext(0, &deferredContext);
    if (FAILED(hr)) {
      return {static_cast<int>(hr), "Failed to create deferred context",
              "D3D11_NVIDIA_DRIVER_COMMAND_LIST_CORRUPTION"};
    }

    deferredContexts.push_back(deferredContext);

    // Record commands in deferred context
    deferredContext->ClearState();

    // Add invalid operations
    deferredContext->Draw(1000000, 0);           // Excessive vertex count
    deferredContext->DrawIndexed(1000000, 0, 0); // Excessive index count

    // Finish command list
    ComPtr<ID3D11CommandList> commandList;
    hr = deferredContext->FinishCommandList(FALSE, &commandList);
    if (FAILED(hr)) {
      return {static_cast<int>(hr), "Command list corruption",
              "D3D11_NVIDIA_DRIVER_COMMAND_LIST_CORRUPTION"};
    }

    commandLists.push_back(commandList);
  }

  // Execute all command lists simultaneously
  for (auto &cmdList : commandLists) {
    context->ExecuteCommandList(cmdList.Get(), FALSE);
  }

  // Force immediate execution
  context->Flush();

  HRESULT deviceResult = device->GetDeviceRemovedReason();
  if (FAILED(deviceResult)) {
    return {static_cast<int>(deviceResult), "Command list corruption",
            "D3D11_NVIDIA_DRIVER_COMMAND_LIST_CORRUPTION"};
  }

  return {0, "Success", "NONE"};
}

DirectXError D3D11Renderer::test_nvidia_driver_excessive_draw_calls() {
  // Generate excessive draw calls to overwhelm NVIDIA driver

  // Set up basic rendering state
  if (vertexBuffer) {
    UINT stride = sizeof(float) * 7;
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, vertexBuffer.GetAddressOf(), &stride,
                                &offset);
  }

  if (vertexShader) {
    context->VSSetShader(vertexShader.Get(), nullptr, 0);
  }

  if (pixelShader) {
    context->PSSetShader(pixelShader.Get(), nullptr, 0);
  }

  if (inputLayout) {
    context->IASetInputLayout(inputLayout.Get());
  }

  context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  // Issue massive number of draw calls
  for (int i = 0; i < 1000000; i++) {
    context->Draw(3, 0);

    // Also issue other GPU commands to increase pressure
    if (i % 1000 == 0) {
      context->Flush();

      // Check for device removal every so often
      HRESULT deviceResult = device->GetDeviceRemovedReason();
      if (FAILED(deviceResult)) {
        return {static_cast<int>(deviceResult), "Excessive draw calls",
                "D3D11_NVIDIA_DRIVER_EXCESSIVE_DRAW_CALLS"};
      }
    }
  }

  // Final flush and check
  context->Flush();
  HRESULT deviceResult = device->GetDeviceRemovedReason();
  if (FAILED(deviceResult)) {
    return {static_cast<int>(deviceResult), "Excessive draw calls",
            "D3D11_NVIDIA_DRIVER_EXCESSIVE_DRAW_CALLS"};
  }

  return {0, "Success", "NONE"};
}

DirectXError
D3D11Renderer::test_nvidia_driver_concurrent_context_destruction() {
  // This is extremely dangerous - concurrent context operations

  // Create a secondary device context
  ComPtr<ID3D11DeviceContext> secondaryContext;
  HRESULT hr = device->CreateDeferredContext(0, &secondaryContext);
  if (FAILED(hr)) {
    return {static_cast<int>(hr), "Failed to create secondary context",
            "D3D11_NVIDIA_DRIVER_CONCURRENT_CONTEXT_DESTRUCTION"};
  }

  // Start operations on secondary context
  secondaryContext->ClearState();

  // Create a long-running operation
  for (int i = 0; i < 10000; i++) {
    secondaryContext->Draw(3, 0);
  }

  ComPtr<ID3D11CommandList> commandList;
  hr = secondaryContext->FinishCommandList(FALSE, &commandList);
  if (FAILED(hr)) {
    return {static_cast<int>(hr), "Failed to finish command list",
            "D3D11_NVIDIA_DRIVER_CONCURRENT_CONTEXT_DESTRUCTION"};
  }

  // Execute command list on main context
  context->ExecuteCommandList(commandList.Get(), FALSE);

  // Immediately release secondary context while work might still be executing
  secondaryContext.Reset();
  commandList.Reset();

  // Force device reset while work is potentially running (DANGEROUS!)
  // Note: This is extremely dangerous and may crash the entire system

  // Clear the current context to force driver state changes
  context->ClearState();
  context->Flush();

  HRESULT deviceResult = device->GetDeviceRemovedReason();
  if (FAILED(deviceResult)) {
    return {static_cast<int>(deviceResult), "Concurrent context destruction",
            "D3D11_NVIDIA_DRIVER_CONCURRENT_CONTEXT_DESTRUCTION"};
  }

  return {0, "Success", "NONE"};
}

#endif // _WIN32