This is a stool classifier for Auggi (http://auggi.ai).  It predicts the Bristol Stool Scale value of a stool image (https://en.wikipedia.org/wiki/Bristol_stool_scale).

The stool image is first segmented using a fine-tuned SegNet, and then a bounding box is calculated from the binary mask to localize the stool.  Once localized, the region of interest is classified using a fine-tuned ResNet18.

The ResNet classifier was trained using low-shot learning techniques to account for the highly imbalanced dataset.



