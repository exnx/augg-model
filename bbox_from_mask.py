import numpy as np
import imutils
import cv2
    
class BoundingBox:

    # def denormalize(self, img):

    #     img = img.astype(float) # need this

    #     img[0] = img[0] * 0.229
    #     img[1] = img[1] * 0.224 
    #     img[2] = img[2] * 0.225 
    #     img[0] += 0.485 
    #     img[1] += 0.456 
    #     img[2] += 0.406

    #     return img

    def get_box(self, rgb_img, mask_img, display=False):

        '''

        # find area of each contour, and all contours
        # if area of contour is greater than threshold, include it in list
        # track min and max of x, y over all the tracked contours

        '''

        # find contours
        img_height, img_width, img_channels = mask_img.shape  # get image shape
        _, cnts, hierarchy = cv2.findContours(mask_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # track total area
        total_area = 0
        valid_contours = []
        THRESHOLD = 0.05  # threshold % for small components to be considered in bounding box

        # loop through all contours to calc total stool area of connect comp.
        for cnt in cnts:
            curr_area = cv2.contourArea(cnt)  # calc cnt area
            total_area += curr_area  # acc area

        # track bbox coordinates of largest bounding box
        x_min = None
        y_min = None
        x_max = None
        y_max = None

        # loop through all cnts
        for cnt in cnts:
            # get area of current contour
            curr_area = cv2.contourArea(cnt)
            # if area is greater than threshold
            if curr_area / total_area > THRESHOLD:

                # get the rectangle values current contour (xc, yc)
                xc_min, yc_min, wc, hc = cv2.boundingRect(cnt)
                
                # calc the max coords
                xc_max = xc_min + wc
                yc_max = yc_min + hc

                # if curr bounding box is beyond the bounds of the largest bbox, replace it
                if x_min is None or xc_min < x_min:
                    x_min = xc_min
                if y_min is None or yc_min < y_min:
                    y_min = yc_min
                if x_max is None or xc_max > x_max:
                    x_max = xc_max
                if y_max is None or yc_max > y_max:
                    y_max = yc_max

        if x_min:

            # expand the bounding box by a margin, but not beyond image bounds
            MARGIN = 0.1
            bbox_height = y_max - y_min  # curr bounding height
            bbox_width = x_max - x_min  # curr bounding width
            height_margin = int(bbox_height * MARGIN / 2)  # margin to add on both ends of bbox
            width_margin = int(bbox_width * MARGIN / 2)  # margin to add on both ends of bbox

            # add margin to height and width on both ends, but clamp within the dimensions of image
            x_min = max(x_min - width_margin, 0)
            y_min = max(y_min - height_margin, 0)
            x_max = min(x_max + width_margin, img_width)
            y_max = min(y_max + height_margin, img_height)

            # # display
            # if display and x_min:

            #     display_img = rgb_img

            #     # if SHOW_RGB:
            #     #     mask_img = rgb_img

            #     # for display
            #     # img_display = cv2.bitwise_not(mask_img)  # invert the image, white background instead

            #     # convert back to rgb
            #     # img_display = cv2.cvtColor(img_display,cv2.COLOR_GRAY2RGB)
            #     # cv2.rectangle(img_display, (x_min,y_min),(x_max, y_max),(255,0,0),6)  # draw the bbox

            #     cv2.rectangle(display_img, (x_min,y_min),(x_max, y_max),(255,0,0),3)  # draw the bbox

            #     # resize the image to display smaller but undistorted
            #     NEW_HEIGHT = 600
            #     ratio = NEW_HEIGHT / img_height
            #     dim = (int(img_width * ratio) , NEW_HEIGHT)
            #     display_img = cv2.resize(display_img, dim)   

            #     # rgb_img = self.denormalize(rgb_img)

            #     cv2.imshow("localized stool", display_img)
            #     cv2.waitKey(0)

        return (x_min, y_min, x_max, y_max)

# for testing purposes
def main():
    
    img_path = 'data/auggi/train/mask_ee8254e0-cee7-4afb-a804-b619dd949bf2.png'

    box = BoundingBox()
    box.get_box(img_path, True)
    
if __name__ == '__main__':
    main()

