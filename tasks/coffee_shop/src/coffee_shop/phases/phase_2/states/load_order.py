#!/usr/bin/env python3
import smach
import numpy as np
from play_motion_msgs.msg import PlayMotionGoal
import json

class LoadOrder(smach.State):

    def __init__(self, context):
        smach.StateMachine.__init__(self, outcomes=['done'])
        self.context = context

    def execute(self, userdata):
        self.context.base_controller.rotate(np.pi)
        pm_goal = PlayMotionGoal(motion_name="load_unload", skip_planning=True)
        self.context.play_motion_client.send_goal_and_wait(pm_goal)
        self.context.voice_controller.sync_tts("Please load the order and say `all done` when you are ready for me to deliver it.")
        while True:
            resp = self.context.speech()
            if not resp.success:
                continue
            resp = json.loads(resp.json_response)
            if resp["intent"]["name"] != "wake_word":
                continue
            if resp["entities"].get("wake", None):
                break
        pm_goal = PlayMotionGoal(motion_name="back_to_default", skip_planning=True)
        self.context.play_motion_client.send_goal_and_wait(pm_goal)
        self.context.start_head_manager("head_manager", '')
        return 'done'