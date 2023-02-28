
import std.stdio;
import std.process;
import std.datetime.stopwatch : StopWatch;
import std.conv : to;

import dcv.core;
import dcv.imgproc;
import dcv.plot;

import bindbc.onnxruntime;

import mir.ndslice;

// compile release for speed: dub -b release
// Video resolution
enum W = 640;
enum H = 480;

void main()
{
    // for video file as input
    auto pipes = pipeProcess(["ffmpeg", "-y", "-hwaccel", "auto", "-i", "pexels-tim-samuel-5834623.mp4", "-vf", "scale=640:480", "-r", "18", "-f", "image2pipe",
     "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"], // yuv420p
        Redirect.stdout);
    // for camera device as input
    /*auto pipes = pipeProcess(["ffmpeg", "-y", "-hwaccel", "auto", "-f", "dshow", "-video_size", "640x480", "-i",
        `video=Lenovo EasyCamera`, "-framerate", "30", "-f", "image2pipe", "-vcodec", "rawvideo", 
        "-pix_fmt", "rgb24", "-"], Redirect.stdout);*/
    
    auto font = TtfFont(cast(ubyte[])import("Nunito-Regular.ttf"));
    
    auto frame = slice!ubyte([H, W, 3], 0);

    auto fig = imshow(frame, "detection");

    int waitFrame = 1;
    StopWatch s;

    //////////////////// init model ///////////////////
    const support = loadONNXRuntime();
    if (support == ONNXRuntimeSupport.noLibrary /*|| support == ONNXRuntimeSupport.badLibrary*/)
    {
        writeln("Please download library from https://github.com/microsoft/onnxruntime/releases");
        return;
    }

    const(OrtApi)* ort = OrtGetApiBase().GetApi(ORT_API_VERSION);
    assert(ort);

    void checkStatus(OrtStatus* status)
    {
        if (status)
        {
            auto msg = ort.GetErrorMessage(status).to!string();
            stderr.writeln(msg);
            ort.ReleaseStatus(status);
            throw new Error(msg);
        }
    }

    OrtLoggingLevel LOGlevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR; //OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
    OrtEnv* env;
    checkStatus(ort.CreateEnv(LOGlevel, "test", &env));
    scope (exit)
        ort.ReleaseEnv(env);

    OrtSessionOptions* session_options;
    checkStatus(ort.CreateSessionOptions(&session_options));
    scope (exit)
        ort.ReleaseSessionOptions(session_options);
    ort.SetIntraOpNumThreads(session_options, 4);
    ort.SetSessionLogSeverityLevel(session_options, 4);

    ort.SetSessionGraphOptimizationLevel(session_options, GraphOptimizationLevel.ORT_ENABLE_ALL);
    
    //OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    
    OrtSession* session;
    checkStatus(ort.CreateSession(env, "yolov3-tiny.onnx", session_options, &session));
    scope (exit)
        ort.ReleaseSession(session);
    
    OrtAllocator* allocator;
    checkStatus(ort.GetAllocatorWithDefaultOptions(&allocator));

    size_t num_input_nodes;
    // print number of model input nodes
    checkStatus(ort.SessionGetInputCount(session, &num_input_nodes));

    immutable(char)*[2] input_node_names = ["input_1".ptr, "image_shape".ptr];
    long[2] input_node_dims = [4, 2];
    long[3] output_node_dims = [3, 3, 3];

    OrtMemoryInfo* memory_info;
    checkStatus(ort.CreateCpuMemoryInfo(OrtAllocatorType.OrtArenaAllocator,
            OrtMemType.OrtMemTypeDefault, &memory_info));
    scope (exit)
        ort.ReleaseMemoryInfo(memory_info);
    
    // score model & input tensor, get back output tensor
    const(char)*[3] output_node_names = [ // check those : https://netron.app/
        "yolonms_layer_1", "yolonms_layer_1:1", "yolonms_layer_1:2"];
    /////////////////// init model ends ///////////////

    s.start;

    while(1)
    {
        import std.algorithm.comparison : max;

        s.reset;
        // Read a frame from the input pipe into the buffer
        ubyte[] dt = pipes.stdout.rawRead(frame.ptr[0..H*W*3]);
        // If we didn't get a frame of video, we're probably at the end
        if (dt.length != H*W*3) break;

        fig.draw(frame, ImageFormat.IF_RGB);
        ///////////////// test model ///////////////

		auto impr = preprocess(frame);
        
        OrtValue*[2] input_tensor;

        long[4] in1 = [1, 3, 416, 416];
        size_t input_tensor_size = 416 * 416 * 3;
        checkStatus(ort.CreateTensorWithDataAsOrtValue(
            memory_info, cast(void*)impr.ptr, input_tensor_size * float.sizeof, in1.ptr, 4,
                ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor[0]));

        float[2] shape = [416.0f, 416.0f];
        long[2] idims = [1, 2];
        checkStatus(ort.CreateTensorWithDataAsOrtValue(
            memory_info, cast(void*)shape.ptr, shape.length * float.sizeof, idims.ptr, 2,
                ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor[1]));

        scope (exit)
            ort.ReleaseValue(input_tensor[0]);
        scope (exit)
            ort.ReleaseValue(input_tensor[1]);

        int is_tensor;
        checkStatus(ort.IsTensor(input_tensor[0], &is_tensor));
        assert(is_tensor);

        OrtValue*[3] output_tensors;
        checkStatus(ort.Run(session, null, input_node_names.ptr, input_tensor.ptr, 2,
                output_node_names.ptr, 3, output_tensors.ptr));
        
        scope (exit){
            ort.ReleaseValue(output_tensors[0]);
            ort.ReleaseValue(output_tensors[1]);
            ort.ReleaseValue(output_tensors[2]);
        }

        float* out0;
        checkStatus(ort.GetTensorMutableData(output_tensors[0], cast(void**)&out0));
        checkStatus(ort.IsTensor(output_tensors[0], &is_tensor));
        assert(is_tensor);

        float* out1;
        checkStatus(ort.GetTensorMutableData(output_tensors[1], cast(void**)&out1));
        checkStatus(ort.IsTensor(output_tensors[1], &is_tensor));
        assert(is_tensor);

        int* out2;
        checkStatus(ort.GetTensorMutableData(output_tensors[2], cast(void**)&out2));
        checkStatus(ort.IsTensor(output_tensors[2], &is_tensor));
        assert(is_tensor);

        OrtTensorTypeAndShapeInfo* sh0; scope(exit) ort.ReleaseTensorTypeAndShapeInfo(sh0);
        checkStatus(ort.GetTensorTypeAndShape(output_tensors[0], &sh0));
        size_t ecount0;
        checkStatus(ort.GetTensorShapeElementCount(sh0, &ecount0));
        size_t dcount0;
        checkStatus(ort.GetDimensionsCount(sh0, &dcount0));
        long[] dims0; dims0.length = dcount0;
        checkStatus(ort.GetDimensions(sh0, dims0.ptr, dcount0));
        //writeln(dims0, " -> ", ecount0);

        OrtTensorTypeAndShapeInfo* sh1; scope(exit) ort.ReleaseTensorTypeAndShapeInfo(sh1);
        checkStatus(ort.GetTensorTypeAndShape(output_tensors[1], &sh1));
        size_t ecount1;
        checkStatus(ort.GetTensorShapeElementCount(sh1, &ecount1));
        size_t dcount1;
        checkStatus(ort.GetDimensionsCount(sh1, &dcount1));
        long[] dims1; dims1.length = dcount1;
        checkStatus(ort.GetDimensions(sh1, dims1.ptr, dcount1));
        //writeln(dims1, " -> ", ecount1);

        OrtTensorTypeAndShapeInfo* sh2; scope(exit) ort.ReleaseTensorTypeAndShapeInfo(sh2);
        checkStatus(ort.GetTensorTypeAndShape(output_tensors[2], &sh2));
        size_t ecount2;
        checkStatus(ort.GetTensorShapeElementCount(sh2, &ecount2));
        size_t dcount2;
        checkStatus(ort.GetDimensionsCount(sh2, &dcount2));
        long[] dims2; dims2.length = dcount2;
        checkStatus(ort.GetDimensions(sh2, dims2.ptr, dcount2));
        //writeln(dims2, " -> ", ecount2);
        
        // create slice shells over pointers without extra allocations
        auto boxCoordinates = out0[0..ecount0].sliced(dims0[0], dims0[1], dims0[2]); // Slice!(float*, 3, Contiguous)

        auto scoresOfBoxes = out1[0..ecount1].sliced(dims1[0], dims1[1], dims1[2]); // Slice!(float*, 3, Contiguous)

        auto selectedIndices = out2[0..ecount2].sliced(dims2[0], dims2[1], dims2[2]); // Slice!(int*, 3, Contiguous)

        //long exec_ms = sw.peek.total!"msecs";
        //exec_ms.writeln;

        // input blob also can be used to display like: impr.transposed!(1, 2, 0) * 255;
        foreach(idx_; selectedIndices[0]){
            auto box = boxCoordinates[0][idx_[2]] / scale;
            
            //writeln(idx_[1]);
            fig.drawRectangle([PlotPoint(box[1], box[0]), PlotPoint(box[3], box[2])], plotBlue, 2.0f);
            fig.drawText(font, classNames[idx_[1]], PlotPoint(cast(float)box[1], cast(float)box[0]),
                        0.0f, 30, plotGreen);

        }

        //////////////// test model ///////////////
        
        int wait = max(1, waitFrame - cast(int)s.peek.total!"msecs");
        
        if (waitKey(wait) == KEY_ESCAPE)
            break;

        if (!fig.visible)
            break;
    }
    
    
}

float scale;

auto letterbox_image(InputImg)(InputImg image, size_t h, size_t w){
    import std.algorithm.comparison : min;

    auto iw = image.shape[1];
    auto ih = image.shape[0];
    scale = min((cast(float)w)/iw, (cast(float)h)/ih);
    auto nw = cast(int)(iw*scale);
    auto nh = cast(int)(ih*scale);
    
    auto resized = resize(image, [nh, nw]);
    
    auto new_image = slice!ubyte([h, w, 3], 128);
    new_image[0..nh, 0..nw, 0..$] = resized[0..nh, 0..nw, 0..$];
    
    //imshow(new_image, ImageFormat.IF_RGB); waitKey();
    
    return new_image;
}

auto preprocess(InputImg)(InputImg img){
    auto w = 416;
    auto h = 416;
    auto boxed_image = letterbox_image(img, h, w);
    auto image_data = boxed_image.as!float;
    
    auto image_data_t = (image_data / 255.0f).transposed!(2, 0, 1);
    

    //auto blob = slice!float([1, 3, 416, 416], 0);
    //blob[0, 0..3, 0..416, 0..416] = image_data_t[0..3, 0..416, 0..416];

    //return blob;
    return image_data_t.slice;
}

enum classNames = [
"person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
];