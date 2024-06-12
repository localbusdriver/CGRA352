# CORE Part 2

## bool isWithinAperture()
Checks if a given UV coordinate is within the aperture radius.
(Adds Epsilon to the radius to account for slight floating point inaccuracies.)

**Aperture in UV coordinates**
* Aperture refers to an opening through which light travels.
* The UV coordinates correspond to different angles of light rays entering the camera.
* By pecifying a *centerU*, *centerV* and a *radius* the function creates a circular aperture.
* The aperture is used to select which rays (UV coordinates) contribute to the final onstructed image, based on the position relative to the center of the aperture.

**Floating Point Inaccuracies**
* Floating point inaccuracies occur because the way computers represent real numbers in binary format cannot precisely represent all real numbers. 
* Not all decimal numbers can be represented in binary format which can lead to small errors.
* *Cumulative Errors*: When performing arithmatic (Mathmatic) operations involving decimal numbers, the small inaccuracies can accumulate resulting in a more significant error in the final result.
* Since the euclidean distance formula involves both multiplication and square root operations, it is very susceptible to these inaccuracies. And since the distance is very important for determining whether a point is within the aperture, even a tiny error could impact whether a point is considered inside or outside the aperture.
* *Epsilon*: To mitigate the floating point errors, an 'EPSILON' value is added to the radius when comparing values. The EPSILON serves as a small tolerance that ensures points are theoretically just outside the aperture. 
 * As floating point errors may be cause certain coordinates to be considered outside the aperture the 'EPSILON' value prevents artifacts or inaccurate calculations from affecting the value.



## cv::Mat constructUVImage()
Function to construct UV-image for given ST coordinates with specific aperture.

**Process**
* Iterates over a predefined grid and finds the UV coordinates with an offset (const), of a scaled amount, from the center.
* For each UV coordinate, it checks if it is within the aperture and extracts the corresponding pixel based on the 't' and 's' values and copies them into an image.

**offset** *(when calculating the UV coordinates)* and **Scales**
* Centers the calculation around a specific point in the UV plane (centerU and centerV).


## void generateSTArray()
Creates a larger image using the constructImage() function multiple times and then displays it.



# COMPLETION/CHALLENGE

## cv::Mat refocus()
Performs synthetic refocusing of a light field.

**Color Accumulation** 
* For each pixel, it averages the colors from diff