import numpy as np
from PIL import Image
from math import atan, pi, exp, sqrt
## CANNY ##
K = 3
def visualize_image(image_array, title="Image"):
   if image_array.dtype != "uint8":
       image_array = (255 * (image_array / image_array.max())).astype("uint8")
   image = Image.fromarray(image_array)
   image.show(title=title)
def median_filter(image, filter_size=3):
   if isinstance(image, Image.Image):
       image = np.array(image)
   pad_size = filter_size // 2
   padded_image = np.pad(image, pad_size, mode='edge')
   filtered_image = np.zeros_like(image)
   for i in range(image.shape[0]):
       for j in range(image.shape[1]):
           window = padded_image[i:i+filter_size, j:j+filter_size]
           filtered_image[i, j] = np.median(window)
   return Image.fromarray(filtered_image.astype(np.uint8))
def getWidth(imageArray):
   return len(imageArray[0])
def getHeight(imageArray):
   return len(imageArray)
def convertToGrayScale(imageArray):
   return np.dot(imageArray[..., :3], [0.2126, 0.7152, 0.0722])
def custom_norm_cdf(x):
   """Compute the cumulative distribution function for a standard normal distribution."""
   return 0.5 * (1 + custom_erf(x / sqrt(2)))
def custom_erf(z):
   """Approximate the error function using a numerical method."""
   t = 1 / (1 + 0.3275911 * abs(z))
   a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
   erf = 1 - (a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5) * exp(-z**2)
   return erf if z >= 0 else -erf
def gkern(kernlen=21, nsig=1):
   interval = (2 * nsig + 1.) / kernlen
   x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
   kern1d = np.diff([custom_norm_cdf(xi) for xi in x])
   kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
   kernel = kernel_raw / kernel_raw.sum()
   return kernel
def getSx():
   return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
def getSy():
   return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
def filter(kernel, imageArray, k):
   width = getWidth(imageArray)
   height = getHeight(imageArray)
   result = np.empty([height - 2*k, width - 2*k])
   for x in range(k, width - k):
       for y in range(k, height - k):
           region = imageArray[y-k:y+k+1, x-k:x+k+1]
           result[y-k, x-k] = np.sum(kernel * region)
   return result
def calculateMagnituteAndDegree(X, Y):
   mag = np.sqrt(X**2 + Y**2)
   degree = np.arctan2(Y, X) * 180 / pi
   return mag, degree
def nonMaximalSupress(image, gdegree):
   width, height = image.shape[1], image.shape[0]
   suppressed_image = np.zeros_like(image)
   for x in range(1, width - 1):
       for y in range(1, height - 1):
           direction = gdegree[y, x] % 180
           if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
               neighbors = [image[y, x-1], image[y, x+1]]
           elif 22.5 <= direction < 67.5:
               neighbors = [image[y-1, x+1], image[y+1, x-1]]
           elif 67.5 <= direction < 112.5:
               neighbors = [image[y-1, x], image[y+1, x]]
           else:
               neighbors = [image[y-1, x-1], image[y+1, x+1]]
           if image[y, x] >= max(neighbors):
               suppressed_image[y, x] = image[y, x]
   return suppressed_image
def doubleThreshold(image, lowThreshold, highThreshold):
   strong = 255
   weak = 75
   result = np.zeros_like(image)
   strong_coords = np.where(image >= highThreshold)
   weak_coords = np.where((image >= lowThreshold) & (image < highThreshold))
   result[strong_coords] = strong
   result[weak_coords] = weak
   return result
def edgeTracking(image):
   strong = 255
   weak = 75
   padded_image = np.pad(image, 1, mode='constant')
   for i in range(1, image.shape[0] + 1):
       for j in range(1, image.shape[1] + 1):
           if padded_image[i, j] == weak:
               if np.any(padded_image[i-1:i+2, j-1:j+2] == strong):
                   padded_image[i, j] = strong
               else:
                   padded_image[i, j] = 0
   return padded_image[1:-1, 1:-1]
def myCannyEdgeDetector(src, sigma, lowThreshold, highThreshold):
   image = Image.open(src).convert("RGB")
   pixel = np.array(image)
   if len(pixel.shape) == 3:
       output = convertToGrayScale(pixel)
   else:
       output = pixel
   output = filter(gkern(2*K+1, sigma), output, K)
   sx = filter(getSx(), output, 1)
   sy = filter(getSy(), output, 1)
   gradientMagnitute, gradientdegree = calculateMagnituteAndDegree(sx, sy)
   suppressed = nonMaximalSupress(gradientMagnitute, gradientdegree)
   thresholded = doubleThreshold(suppressed, lowThreshold, highThreshold)
   output = edgeTracking(thresholded)
   #visualize_image(output)
   return output
# Import or define your custom Canny Edge Detector
def preprocess_image(image_path):
   """Preprocess the image: grayscale, edge detection."""
   # Open the image
   image = Image.open(image_path)
   # Convert to RGB mode if the image has an alpha channel (e.g., RGBA)
   if image.mode == "RGBA":
       image = image.convert("RGB")
   # Perform edge detection
   edges = myCannyEdgeDetector(image_path, sigma=1, lowThreshold=50, highThreshold=150)
   return edges
def resize_image(image_path):
   image = Image.open(image_path)
   image = image.resize((250, 250))
   image.save(image_path)  # Save resized image to the same path
   return image
def custom_extract_contours(image):
   """
   Extract contours from the edge-detected image manually.
   :param image: Edge-detected binary image (NumPy array).
   :return: List of contours, each contour is a list of (x, y) points.
   """
   height, width = image.shape
   visited = np.zeros_like(image, dtype=bool)
   contours = []
   def dfs(x, y, contour):
       """Depth-first search to trace a contour."""
       stack = [(x, y)]
       while stack:
           cx, cy = stack.pop()
           if 0 <= cx < width and 0 <= cy < height and image[cy, cx] > 0 and not visited[cy, cx]:
               visited[cy, cx] = True
               contour.append((cx, cy))
               # Explore 8-connected neighbors
               stack.extend([
                   (cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1),
                   (cx + 1, cy + 1), (cx - 1, cy - 1), (cx + 1, cy - 1), (cx - 1, cy + 1)
               ])
   for y in range(height):
       for x in range(width):
           if image[y, x] > 0 and not visited[y, x]:
               contour = []
               dfs(x, y, contour)
               if len(contour) > 10:  # Ignore small noise contours
                   contours.append(contour)
   return contours
def custom_match_sign(input_image_path, templates, n=5, weights=[10, 5, 6, 2, 1]):
   """
   Match input image with template images using custom contour similarity.
   Apply custom weights to give more importance to larger contours.
   :param input_image_path: Path to the input image.
   :param templates: Dictionary of template names and paths.
   :param n: Number of largest contours to compare.
   :param weights: List of weights for the top `n` contours.
   :return: Name of the best matching traffic sign.
   """
   # Validate weights
   if len(weights) != n:
       raise ValueError(f"Length of weights must match the number of contours (n={n}).")
   input_edges = preprocess_image(input_image_path)
   input_contours = custom_extract_contours(input_edges)
   # Sort input contours by size (descending order)
   input_contours = sorted(input_contours, key=len, reverse=True)[:n]
   best_match = None
   lowest_match_value = float("inf")
   for sign_name, template_path in templates.items():
       template_edges = preprocess_image(template_path)
       template_contours = custom_extract_contours(template_edges)
       # Sort template contours by size (descending order)
       template_contours = sorted(template_contours, key=len, reverse=True)[:n]
       if input_contours and template_contours:
           # Blank images for visualizing compared contours
           input_visual = np.zeros_like(input_edges, dtype=np.uint8)
           template_visual = np.zeros_like(template_edges, dtype=np.uint8)
           # Draw only the top n contours on the blank images
           for input_contour, template_contour in zip(input_contours, template_contours):
               for x, y in input_contour:
                   input_visual[y, x] = 255  # Draw the input contour
               for x, y in template_contour:
                   template_visual[y, x] = 255  # Draw the template contour
           # TEMPORARY VISUALIZATION ARRAYS FOR DEBUGGING/COMPARISON
           # Create temporary copies for visualization
           input_visual_temp = input_visual.copy()
           template_visual_temp = template_visual.copy()
           # Match dimensions by padding the temporary arrays
           height = max(input_visual_temp.shape[0], template_visual_temp.shape[0])
           input_visual_resized = np.pad(input_visual_temp, ((0, height - input_visual_temp.shape[0]), (0, 0)), mode='constant', constant_values=0)
           template_visual_resized = np.pad(template_visual_temp, ((0, height - template_visual_temp.shape[0]), (0, 0)), mode='constant', constant_values=0)
           # Combine the temporary visuals for comparison
           combined_visual = np.hstack((input_visual_resized, template_visual_resized))
           # Visualize the combined temporary arrays
           visualize_image(combined_visual, title=f"Compared Contours: {sign_name}")
           # Calculate weighted similarity
           total_match_value = 0
           for i, (input_contour, template_contour) in enumerate(zip(input_contours, template_contours)):
               # Normalize the contours (scale and center)
               def normalize_contour(contour):
                   contour = np.array(contour)
                   center = contour.mean(axis=0)
                   norm_contour = contour - center
                   scale = np.linalg.norm(norm_contour, axis=1).max()
                   return norm_contour / scale
               input_normalized = normalize_contour(input_contour)
               template_normalized = normalize_contour(template_contour)
               # Calculate similarity (e.g., mean squared distance)
               match_value = np.mean([
                   np.min(np.linalg.norm(input_normalized - pt, axis=1))
                   for pt in template_normalized
               ])
               # Apply weight to the match value
               weighted_match_value = match_value * weights[i]
               total_match_value += weighted_match_value
           # Compute the weighted match value for this template
           avg_match_value = total_match_value
           print(f"Match value with {sign_name}: {avg_match_value}")
           if avg_match_value < lowest_match_value:
               lowest_match_value = avg_match_value
               best_match = sign_name
   return best_match
# Define the traffic sign templates
templates = {
        "STOP": "../templates/stop.png",
        "YIELD": "../templates/yield.webp",
        "SPEED LIMIT": "../templates/speed_limit.jpg",
        "HOSPITAL": "../templates/hospital.png",
        "NO ENTRY": "../templates/noentry.webp",
        "TWO WAY": "../templates/two-way.jpg",
        "PEDESTRIAN CROSSING": "../templates/pedestrian.jpg",
        "ROAD NARROWS AHEAD": "../templates/narrow.webp",
        "RAIL CROSSING": "../templates/rail_crossing.webp",
        "NO U TURN ALLOWED": "../templates/nouturn.webp",
        "SNOW WARNING": "../templates/snow_warning.webp",
        "DEAD END": "../templates/dead_end.webp",
        "CYCLISTS NOT PERMITTED": "../templates/no_cycle.webp",
        "END OF THE OVERTAKING PROHIBITION": "../templates/end_overtaking.png",
        "ROAD WORK": "../templates/road_work.png",
        "SLIPPERY ROAD AHEAD": "../templates/slippery_road.webp",
    }
# Input image to identify
input_image_path = "../tests/slippery1.jpg"
# Identify the traffic sign
identified_sign = custom_match_sign(input_image_path, templates)
print(f"Identified Traffic Sign: {identified_sign}")