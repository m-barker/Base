import smach
import rospy
from std_msgs.msg import Int16
from tiago_controllers.helpers.pose_helpers import get_pose_from_param


class StartPhase3(smach.State):
    def __init__(self, default):
        smach.State.__init__(self, outcomes=["success"])
        self.default = default

    def execute(self, userdata):
        self.default.voice.speak("A final update - I am starting Phase 3.")

        res = self.default.controllers.base_controller.sync_to_pose(
            get_pose_from_param("/wait/pose")
        )

        # self.default.datahub_start_phase.publish(Int16(3))

        return "success"
