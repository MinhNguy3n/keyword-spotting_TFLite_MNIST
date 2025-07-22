/* MNIST classifier
This example use the hand-written digit recongition model that was pruned and 
quantized to classify example converted test samples digits (sample_x.h)
==============================================================================*/

#include <TensorFlowLite.h>
// #include "main_functions.h"

#include "MNIST_quantized_int8_test.h" // model data
#include "sample_0.h"
#include "sample_1.h" 
#include "sample_2.h" 
#include "sample_3.h" 
#include "sample_4.h" 
#include "sample_5.h" 
#include "sample_6.h" 
#include "sample_7.h" 
#include "sample_8.h"     
#include "sample_9.h"
#include "sample_benchmark.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // if we want to use only specific operations from micro_op_resolver 
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "model_setting.h"
#include <strings.h>
#include <stdlib.h>

// #define MNIST_FLOAT
#define MNIST_INT8

// Globals, used for compatibility with Arduino-style sketches.

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
// tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 100*1024;
// byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));
alignas(16) uint8_t tensorArena[tensorArenaSize];

/**
 The name of this function is important for Arduino compatibility.
 **/
void setup() {
  
  tflite::InitializeTarget();
  // Serial.println("Starting program...");
  MicroPrintf("Starting program...");

    // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(g_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    // Serial.println("Model schema version not supported!");
    MicroPrintf(
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      tflModel->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  // Serial.println("Read MNIST model");
  MicroPrintf("Read MNIST model: OK");

  static tflite::AllOpsResolver Tflresolver;

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, Tflresolver, tensorArena, tensorArenaSize);
  // Serial.println("Created TFLite interpreter");
  // Allocate memory for the model's input and output tensors
  TfLiteStatus allocationStatus = tflInterpreter->AllocateTensors();
  if(allocationStatus != kTfLiteOk){
    // Serial.println("Tensor Allocation failed!");
    MicroPrintf("Tensor Allocation failed!");
  }
  else{
    // Serial.println("Allocated tensors");
    MicroPrintf("Allocated tensors");
  }

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Assertion of input tensor dimensions and data type
  if ((tflInputTensor->dims->size != 3) || (tflInputTensor->dims->data[0] != 1) ||
      (tflInputTensor->dims->data[1] != kNumRows) ||
      (tflInputTensor->dims->data[2] != kNumCols) ||
  #ifdef MNIST_INT8
      (tflInputTensor->type != kTfLiteUInt8)) {
  #else
      (tflInputTensor->type != kTfLiteFloat32)) {
  #endif
    MicroPrintf("Bad input tensor parameters in model");

    return;
  }
      MicroPrintf("input tensor dims->size = %d. \n"
                "input tensor dims->data[0] = %d. \n" 
                "input tensor dims->data[1] = %d. \n"
                "input tensor dims->data[2] = %d. \n",
                tflInputTensor->dims->size, tflInputTensor->dims->data[0],
                tflInputTensor->dims->data[1], tflInputTensor->dims->data[2]);
}
/**
This function handle the execution of inference
p:  uint8_t * pointer to sample image array
label: int true label of the sample image
file: char * name of the tested file
**/
void run_inference(const uint8_t * image_data, int label, const char * file) {
  MicroPrintf("Reading tensor data... \n");
  // MicroPrintf("Sample normalized = { \n");
#ifdef MNIST_INT8
  uint8_t * input_data_uint8 = tflInputTensor->data.uint8; // pointer to input tensor as INT8
#else
  float * input_data_float = tflInputTensor->data.f; // pointer to input tensor as FLOAT32
#endif
  const uint8_t* p = image_data; // pointer to individual pixel in image_data array

  // loop over image_data
  for (int col = 0; col < kNumCols; col++){
    for (int row = 0; row < kNumRows; row++, p++){
      // convert pixel value to the quantization scheme of the model's input tensor
#ifdef MNIST_INT8
      *input_data_uint8++ = tflite::FloatToQuantizedType<uint8_t>(p[0] / 255.0f,tflInputTensor->params.scale, tflInputTensor->params.zero_point);
      // *input_data_uint8++ = p[0];
#else
      *input_data_float++ = tflite::FloatToQuantizedType<float>(p[0] / 255.0f,tflInputTensor->params.scale, tflInputTensor->params.zero_point);
#endif
    }
  }

  MicroPrintf("Finished reading pixel data.");

  MicroPrintf("Running Inference ...");
  
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  
  if (invokeStatus != kTfLiteOk) {
    MicroPrintf("Invoke Failed!");
    return;
  }
#ifdef MNIST_INT8
  uint8_t digit_score_uint8;
#else
  float digit_score_float;
#endif
  float digit_score_fraction;
  float digit_score_int;
  float max_digit_score = 0;
  int prediction = 0;
  
  // Loop through the output tensor 
  for (int i = 0; i < kCategoryCount; i++) {
    // save the highest score
#ifdef MNIST_INT8
    digit_score_uint8 = tflOutputTensor->data.uint8[i];
    if(max_digit_score < digit_score_uint8){
      max_digit_score = digit_score_uint8;
      prediction = i;
    }
    float digit_score_float = static_cast<float>((digit_score_uint8 - tflOutputTensor->params.zero_point) * tflOutputTensor->params.scale);
    // get fractional and integer part of the scaled output
    digit_score_fraction = std::modf(digit_score_float*100, &digit_score_int);

#else
    digit_score_float = tflOutputTensor->data.f[i];
    if(max_digit_score < digit_score_float){
      max_digit_score = digit_score_float;
      prediction = i;
    }
    // scale output tensor ([0, 1]f)
    digit_score_float = (digit_score_float - tflOutputTensor->params.zero_point) * tflOutputTensor->params.scale;
    // get fractional and integer part of the scaled output
    digit_score_fraction = std::modf(digit_score_float*100, &digit_score_int);
#endif
    
    MicroPrintf("%s : %d.%d%%(%d)", kCategoryLabels[i], 
      static_cast<int>(digit_score_int), static_cast<int>(digit_score_fraction*1000), digit_score_uint8);
  }
  // print the predicted digit (max(digit_score))
  MicroPrintf("Message file name: %s\nTrue: %d, Prediction = %d", file, label, prediction);
}

/** 
loop: The name of this function is important for Arduino compatibility.
**/
void loop() {
  run_inference(image_data_zero, 0 , "sample_0.h");
  delay(5000);
  run_inference(image_data_one, 1, "sample_1.h");
  delay(5000);
  run_inference(image_data_two, 2, "sample_2.h");
  delay(5000);
  run_inference(image_data_three, 3, "sample_3.h");
  delay(5000);
  run_inference(image_data_four, 4, "sample_4.h");
  delay(5000);
  run_inference(image_data_five, 5, "sample_5.h");
  delay(5000);
  run_inference(image_data_six, 6, "sample_6.h");
  delay(5000);
  run_inference(image_data_seven, 7, "sample_7.h");
  delay(5000);
  run_inference(image_data_eight, 8, "sample_8.h");
  delay(5000);
  run_inference(image_data_nine, 9, "sample_9.h");
  delay(5000);
  run_inference(image_data, 2, "sample_benchmark.h");
  delay(5000);
}
