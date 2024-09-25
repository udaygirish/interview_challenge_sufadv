# Transformers, Robots in disguise
---
### Experimental implementation of Body  Transformer

    C. Sferrazza, D.-M. Huang, F. Liu, J. Lee, and P. Abbeel, ‘Body Transformer: Leveraging Robot Embodiment for Policy Learning’, arXiv [cs.RO]. 2024.
---
### Running the code

Please visit RUN_INSTRUCTIONS.md for the Instructions 
* The code is well documented to understand every component.


---
### BONUS 

#### Question:
The graph for this robot will be each joint connecting to the next (`0 -> 1 -> 2 -> ...`)
and the gripper being connected to all of them (bonus points if you can reason about why this does/doesn't make sense).


#### Answer:
1. The Graph for the Robot joints connecting to the next sequentially does make sense as I'm assuming for a Manipulator each joint is dependent on previous position and it makes total sense to have the sequential configuration.
2. The Gripper being connected to all of them is kind of neutral sense for me atleast. Assuming it is a parallel jaw gripper this configuration means the open or closed state of the gripper is influenced by all the joints. 

To argue: 

* #### When/Why it makes Sense ? 
May be in cases where we need Global influence of all the joints it is needed to have this configuration. Where every joint kind of influences the gripper. 
Not sure exactly where this can be used but I'm assuming complex manipulation or sensitive task based grasping might need this.
Especially when let us take u want to hold a tool in Welding at an angle where safety is important or let us take where a operation can only take 
place at a particular configuration, then we might need the gripper opening
or close depending on whole body configuration in that case having the gripper
attending to whole body makes sense. Also for precise applications where 
a particular overall robot configuration needs to be checked before executing 
gripper commands this can come in help.
But this adds a bit of complexity to the whole pipeline especially on training
stability.

* #### When / Why it does not makes sense ? 
For Simpler applications this simply does not makes sense except overcomplicating the pipeline. So In Simpler robotic systems or simple robot works such as Pick and Place as this does not make much sense with respect to the complexity it adds. Especially, in these case as the last joint is already attending back to the sequential joints before adding gripper alone to last joint might suffice.



### CREDITS

* Paper Implementation
* There are some places where I struggled a bit with shapes and the code for Masked Attention  especially from scratch, Wanted to mention I took help of Claude AI.
* Also used CoPilot for AutoGeneration of Initial Comments and made edits on them. - Easy and efficient - As I  like a well documented code