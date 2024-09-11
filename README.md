# IACV-project

 Project developed by Paolo Riva, Michelangelo Stasi, Mihai-Viorel Grecu c/o Politecnico di Milano
 Course: Image Analysis and Computer Vison - A.A. 2023/24
 This Computer Vision project aims at detecting data in a tennis match through the widely-used Human Pose Detection method and the TRACE methon for ball detection.

 The program focuses specifically on the following tasks:
 0. Identify the field lines and, knowing the field measures, find yhe homography H from field to image.
 1. Use the well-known Human Pose Estimation  method (based on Deep Learning) to identify the articulated segments of the player.
 2. Select the feet (end points of the leg segments) and their position Pleft and Pright in each image
 3. Check whether the feet are static or they are moving (by checking if H^-1 Pleft and/or H^-1 Pright are constant along a short sequence).
    If a foot is static, assume that it is placed on the ground.
 4. Collect the time-sequence of the step points: i.e., the positions H^-1P of the feet in the instances when they were static.
 5. In parallel, try to select the time instants when the player hits the ball with the rackets, and try to compute statistics on the short runs between consecutive hits
 
# To-install
pip install -r requirements.txt

# To-run
*/main.py [video_path.mp4]