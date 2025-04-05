#ifndef TEST_CUH
#define TEST_CUH

struct Frame {
    int width;
    int height;
    uint32_t* pixels;
};

constexpr int DISPLAY_WIDTH = 600;
constexpr int DISPLAY_HEIGHT = 400;


void render(struct Frame*);
void prepare_objects();


#endif //TEST_CUH
