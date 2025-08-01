# logo-detection-tracking

Logo detection and tracking system for videos using DETR and object tracking algorithms

## Key Notes =>

### Problem Analysis

**Objective**: Detect + Track logo (unique id) in video, stream  
**Provide**: samples of logo (one per each, total: 3)  
**Require**: low latency → hard to use LLMs, VLMs  
**Test video** as timelapsed format → requires good tracking modules:
- Large Inter-frame Displacements  
- Illumination Changes  
- Occlusions  
- Appearance Changes (maybe hidden in next frame)  

No provided dataset → need to manually prepare or use zero-shot, one-shot method  
Logo can be augmented (rotate in z-axis, flatten as vertical string, …) in test video  

### Pipeline:

**Object detection (Logo)** → **Recognize Logo** → **Tracking**

#### Object detection (Logo):
- DETR, all logo as 1 class  
  *(Reason: detecting 1 class is better than multiclass, and if detecting out-of-domain logos is needed, we’ll need to collect data + retrain → using 1 class reduces retraining effort)*

#### Recognize Logo (consider using 3 methods sequentially, or concat into 1 vector):

- Reference sample, RoI of obj → CNN → embeddings → compare similarity (model-based ~ Siamese network)  
  *(Reason: leverage image features of logos), e.g., McDonald*  
  - https://github.com/cjvargasc/JNN_recog  
  - https://github.com/cjvargasc/oneshot_siamese  

- Template and Feature Matching (model-free):  
  *(Reason: signs at angles are geometrically distorted, e.g., Disney)*  
  - Quality aware template matching: https://github.com/kamata1729/QATM_pytorch/tree/master  
  - Feature Matching: SIFT/SURF/ORB + Homography (OpenCV-only)

- OCR: If text in RoI ~ text of reference sample → pass  
  *(Reason: most signs/logos contain text, even when placed vertically or with different backgrounds, still detectable, e.g., Ernst & Young)*

#### Tracking and ReID (Include ReID since obj may disappear/occlude in some frames and reappear):
- SORT, DeepSORT, BoT-SORT, ByteTrack *(priority to test because they support ReID, and unique ID is required)*  
- Correlation Filters (e.g., KCF, CSRT)  
- Kalman Filters and Particle Filters  
- Spectral Time-Lapse (STL)

### Optimization for inference:

- Model format conversion: ONNX, TensorRT  
- Model quantization (on weights): weight pruning, data type conversion  
- Searching algorithms: elastic, .. *(if number of logos to detect increases → need fast sim search instead of sequential comparison)*

---

## Test Video Comments:

- Very fast video, many logos and various colors  
- Nighttime and glare from lights  
- Logos are occluded during the video  
- Ernst & Young in video is actually a signboard with vertically arranged text, unlike test image  
- Disney and McDonald have matching test image and video logos  
- Multiple same logos in one frame → need to handle unique tracking  
- McDonald's logo on billboard has red background, different from others on building (no red background)  
- Glass causes logo reflections (Ernst & Young). Some frames have logo reflected on nearby building → difficult for detection  
- Processing time: video is timelapsed → frames removed from original video → implies processing speed is important. If only accuracy was needed, full-frame video would be sent for better detection & tracking.  
- In the requirements, there are inference optimization tasks, including converting detection model to ONNX or TensorRT format → indicates need for lightweight, fast inference model.  
- Core task stated: fine tuning is not required, use modern efficient detection model → need a general solution (not just for 3 logos but scalable if needed)  
- Need optimization plan  
- Need scaling plan (consider multi-processing)
