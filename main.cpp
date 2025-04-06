#define UNICODE
#define _UNICODE
#include <iostream>
#include <ostream>
#include <windows.h>
#include "rendering/rendering.cuh"

static bool quit = false;

Frame frame{};
Camera h_camera;

LRESULT CALLBACK WindowProcessMessage(HWND, UINT, WPARAM, LPARAM);

static BITMAPINFO frame_bitmap_info;
static HBITMAP frame_bitmap = 0;
static HDC frame_device_context = 0;

POINT lastMousePos;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int nCmdShow) {
    AllocConsole(); // Open a new console
    freopen("CONOUT$", "w", stdout); // Redirect stdout to the console

    h_camera.local_matrix = Matrix4x4(
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0}
        ); // * Matrix4x4::rotation_x(15 * (M_PI / 180.0f));
    h_camera.inv_local_matrix = h_camera.local_matrix.inverse();

    prepare_objects();
    update_objects();

    const wchar_t window_class_name[] = L"My Window Class";
    static WNDCLASS window_class = { 0 };
    window_class.lpfnWndProc = WindowProcessMessage;
    window_class.hInstance = hInstance;
    window_class.lpszClassName = window_class_name;
    RegisterClass(&window_class);

    frame_bitmap_info.bmiHeader.biSize = sizeof(frame_bitmap_info.bmiHeader);
    frame_bitmap_info.bmiHeader.biPlanes = 1;
    frame_bitmap_info.bmiHeader.biBitCount = 32;
    frame_bitmap_info.bmiHeader.biCompression = BI_RGB;
    frame_device_context = CreateCompatibleDC(0);

    static HWND window_handle;
    window_handle = CreateWindow(
        window_class_name,
        L"Drawing Pixels",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, NULL, NULL, hInstance, NULL);
    if(window_handle == NULL) { return -1; }

    SetCapture(window_handle);

    while(!quit) {
        static MSG message = { 0 };
        while(PeekMessage(&message, NULL, 0, 0, PM_REMOVE)) { DispatchMessage(&message); }

        render(&frame);

        InvalidateRect(window_handle, NULL, FALSE);
        UpdateWindow(window_handle);
    }

    return 0;
}


LRESULT CALLBACK WindowProcessMessage(HWND window_handle, UINT message, WPARAM wParam, LPARAM lParam) {
    switch(message) {
        case WM_CREATE: {
            SetCapture(window_handle);
            ShowCursor(FALSE);
        } break;

        case WM_QUIT:
        case WM_DESTROY: {
            quit = true;
        } break;

        case WM_MOUSEMOVE: {
            // Get the current cursor position
            RECT rect;
            GetWindowRect(window_handle, &rect); // Get the window's absolute position

            // Calculate the center of the window in screen coordinates
            int centerX = (rect.left + rect.right) / 2;
            int centerY = (rect.top + rect.bottom) / 2;

            POINT currentMousePos;
            GetCursorPos(&currentMousePos);

            // Calculate the movement
            const auto dx = (float)(currentMousePos.x - centerX);
            const auto dy = (float)(currentMousePos.y - centerY);

            auto dir = vec2f {dx, 0};

            if (dir.magnitude() != 0) {
                constexpr float deltaAlpha = .5 * (M_PI / 180);
                vec2f rotation = dir.normalize() * deltaAlpha;

                if (rotation.y != 0) {
                    h_camera.local_matrix = h_camera.local_matrix * Matrix4x4::rotation_y(rotation.y);
                }
                if (rotation.x != 0) {
                    h_camera.local_matrix = h_camera.local_matrix * Matrix4x4::rotation_z(rotation.x);
                }
                h_camera.inv_local_matrix = h_camera.local_matrix.inverse();

                update_objects();
            }

            // Update the last known mouse position
            //lastMousePos = currentMousePos;

            // Set the cursor position back to the center of the window
            SetCursorPos(centerX, centerY);
            break;
        }

        case WM_KEYDOWN: {
            if (wParam == VK_ESCAPE) {
                quit = true;
            }
        }
        case WM_CHAR: {
            if (wParam == 'W') {
                h_camera.local_matrix.translate_origin(vec3f{.1, 0, 0});
                h_camera.inv_local_matrix = h_camera.local_matrix.inverse();

                update_objects();
            } else if (wParam == 'S') {
                h_camera.local_matrix.translate_origin(vec3f{-.1, 0, 0});
                h_camera.inv_local_matrix = h_camera.local_matrix.inverse();

                update_objects();
            } else if (wParam == 'A') {
                h_camera.local_matrix.translate_origin(vec3f{0, -.1, 0});
                h_camera.inv_local_matrix = h_camera.local_matrix.inverse();

                update_objects();
            } else if (wParam == 'D') {
                h_camera.local_matrix.translate_origin(vec3f{0, .1, 0});
                h_camera.inv_local_matrix = h_camera.local_matrix.inverse();

                update_objects();
            }
        }

        case WM_PAINT: {
            static PAINTSTRUCT paint;
            static HDC device_context;
            device_context = BeginPaint(window_handle, &paint);
            BitBlt(device_context,
                   paint.rcPaint.left, paint.rcPaint.top,
                   paint.rcPaint.right - paint.rcPaint.left, paint.rcPaint.bottom - paint.rcPaint.top,
                   frame_device_context,
                   paint.rcPaint.left, paint.rcPaint.top,
                   SRCCOPY);
            EndPaint(window_handle, &paint);
        } break;

        case WM_SIZE: {
            frame_bitmap_info.bmiHeader.biWidth  = LOWORD(lParam);
            frame_bitmap_info.bmiHeader.biHeight = HIWORD(lParam);

            std::cout << frame_bitmap_info.bmiHeader.biWidth << " " << frame_bitmap_info.bmiHeader.biHeight << std::endl;

            if(frame_bitmap) DeleteObject(frame_bitmap);
            frame_bitmap = CreateDIBSection(NULL, &frame_bitmap_info, DIB_RGB_COLORS, (void**)&frame.pixels, 0, 0);
            SelectObject(frame_device_context, frame_bitmap);

            frame.width =  LOWORD(lParam);
            frame.height = HIWORD(lParam);
        } break;

        default: {
            return DefWindowProc(window_handle, message, wParam, lParam);
        }
    }
    return 0;
}