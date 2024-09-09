
import torch
import cv2
import numpy as np
import torch.nn.functional as F

def get_image_contour(image: torch.Tensor) -> (torch.tensor, torch.tensor):
    '''
    Gets the contours of the image , upper-left,lower-left,lower-right,upper-right
    '''
    # Ensure img_tensor is on the CPU and in numpy format C,H,W
    img_numpy = image.detach().cpu().numpy()
    # transpose if needed
    img_numpy = img_numpy.transpose((1, 2, 0))  # H,W,C

    # Convert the image to grayscale
    #gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
    gray = img_numpy.mean(axis=2)
    # Normalize image to [0, 1]
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    # Threshold the grayscale image to create a binary mask for the borders
    _, thresh = cv2.threshold(gray, 1e-3, 1, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        (thresh * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to keep track of the largest contour and its area
    largest_contour = None
    largest_area = 0

    # Iterate through the contours to find the largest one
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Create a black image of the same size as the original image
    result_image = np.zeros_like(gray)

    # Fill the area inside the contours with white
    cv2.fillPoly(result_image, (largest_contour,), 1)
    # Initialize variables to store the positions of the corners
    top_left_position = None
    lower_left_position = None
    lower_right_position = None
    top_right_position = None

    # Iterate through the mask to find the positions of the corners
    for y in range(result_image.shape[0]):
        for x in range(result_image.shape[1]):
            if result_image[y, x] != 0:
                if top_left_position is None:
                    top_left_position = (x, y)
                    lower_left_position = (x, y)
                elif top_left_position is not None:
                    lower_left_position = (x, y)
                    break

    for x in reversed(range(result_image.shape[1])):
        if result_image[top_left_position[1], x] != 0:
            top_right_position = (x, top_left_position[1])
            break

    lower_right_position = (top_right_position[0], lower_left_position[1])

    return torch.tensor([top_left_position, lower_left_position, lower_right_position, top_right_position]), torch.tensor(result_image)


def stretch_image(image: torch.Tensor, contours: torch.Tensor) -> torch.Tensor:
    h_idx, w_idx, c_idx = 1, 2, 0  # [3, H, W]
    cropped_image = image[:, contours[0, 1]
        :contours[1, 1]+1, contours[0, 0]:contours[2, 0]+1]

    # Use torch.nn.functional.interpolate to stretch the image
    stretched_img = F.interpolate(
        cropped_image.unsqueeze(0), size=(image.shape[h_idx], image.shape[w_idx]), mode='bilinear', align_corners=False
    )
    # Remove the batch dimension if you don't need it
    stretched_img = stretched_img.squeeze(0)
    return stretched_img

def stretch_images(images: torch.tensor, masks: torch.tensor) -> (torch.tensor, torch.tensor, list):
    contours_lst = []
    for i in range(images.shape[0]):  # [N, 3, H, W]
        contours, mask = get_image_contour(images[i])
        contours_lst.append(contours)
        masks[i] = mask.unsqueeze(0)
        images[i] = stretch_image(images[i], contours)
    return images, masks, contours_lst

def unstretch_keys(keys: torch.Tensor, contours_lst: list) -> torch.Tensor:
    n_orig, emb_orig, h_orig, w_orig = keys.shape  # N,emb, H, W
    for i in range(keys.shape[0]):  # [N, emb, H, W]
        curr_contours = contours_lst[i]
        target_height = curr_contours[1, 1] - curr_contours[0, 1] + 1
        target_width = curr_contours[2, 0] - curr_contours[0, 0] + 1
        unstretched_tensor = F.interpolate(keys[i].unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False)
        unstretched_tensor = unstretched_tensor.squeeze(0)

        padded_image = torch.zeros(torch.Size([emb_orig, h_orig, w_orig]), device=keys.device)
        padded_image[:, curr_contours[0, 1]:curr_contours[1, 1]+1,
                        curr_contours[0, 0]:curr_contours[2, 0]+1] = unstretched_tensor
        keys[i] = padded_image
    return keys
