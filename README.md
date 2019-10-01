# TTAch
Image Test Time Augmentation with PyTorch!

## Quick start

#####  Segmentation model wrapping:
```python
import ttach as tta
tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
```
#####  Classification model wrapping:
```python
tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crops_transform())
```

## Advanced Examples
#####  Custom transform:
```python
# defined 2 * 2 * 3 * 3 = 36 augmentations !
transform = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
)

tta_model = tta.SegmentationTTAWrapper(model, transform)
```
##### Custom model (multi-input / multi-output)
```python
# Example how to process ONE batch on images with TTA
# Here `image`/`mask` are 4D tensors (B, C, H, W), `label` is 2D tensor (B, N)

for augmenter in transform: # custom transform or e.g. tta.aliases.d4_transform() 
    
    # augment image
    augmented_image = augmenter.augment_image(image)
    
    # pass to model
    model_output = model(augmented_image, another_input_data)
    
    # reverse augmentation for mask and label
    deaug_mask = augmenter.deaugment_mask(model_output['mask'])
    deaug_label = augmenter.deaugment_label(model_output['label'])
    
    # save results
    labels.append(deaug_mask)
    masks.append(deaug_label)
    
# reduce results as you want, e.g mean/max/min
label = mean(labels)
mask = mean(masks)
```
 
## Transforms
 
  - HorizontalFlip()
  - VerticalFlip()
  - Scale(scales=[1, 2, 3], intepolation="nearest")
  - Rotate90(angles=[0, 90, 180, 270])
  - Add(values=[1, 2, 20, -20])
  - Multiply(factors=[0.9, 1, 1.3])
  - FiveCrops(crop_height, crop_width)
 
## Aliases

  - flip_transform
  - hflip_transform
  - d4_transform
  - multiscale_transform
  - five_crop_transform
  - ten_crop_transform

## Run tests

```bash
docker build -f Dockerfile.dev -t ttach:dev . && docker run --rm ttach:dev pytest -p no:cacheprovider
```