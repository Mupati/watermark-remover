const app = new Vue({
  el: "#app",
  delimiters: ["${", "}"],
  data: {
    callPlaced: false,
    localStream: null,
    mutedAudio: false,
    mutedVideo: false,
    onlineUsers: [],
    isLoggedIn: false,
    incomingCall: false,
    incomingCaller: "",
    incomingCallNotification: "",
    rtmClient: null,
    rtmChannel: null,
    rtcClient: null,
    users: [],
    updatedOnlineStatus: {},
    channelName: null,
    isCallingUser: false,
    callingUserNotification: "",
    localAudioTrack: null,
    localVideoTrack: null,
    remoteVideoTrack: null,
    remoteAudioTrack: null,
    resourceId: "",
    recordingSID: "",
    rtcToken: null,
    isRecordingStarted: false,
  },
  mounted() {
    this.fetchUsers();
    this.initRtmInstance();
  },

  created() {
    window.addEventListener("beforeunload", this.logoutUser);
  },

  beforeDestroy() {
    this.endCall();
    this.logoutUser();
  },

  methods: {
    async fetchUsers() {
      const { data } = await axios.get("/users");
      this.users = data;
    },

    async logoutUser() {
      console.log("destroyed!!!");
      this.rtmChannel.leave(AUTH_USER);
      await this.rtmClient.logout();
    },

    async initRtmInstance() {
      // initialize an Agora RTM instance
      this.rtmClient = AgoraRTM.createInstance(AGORA_APP_ID, {
        enableLogUpload: false,
      });

      // RTM Channel to be used
      this.channelName = "videoCallChannel";

      // Generate the RTM token
      const { data } = await this.generateToken(this.channelName);

      // Login when it mounts
      await this.rtmClient.login({
        uid: AUTH_USER,
        token: data.rtm_token,
      });

      this.isLoggedIn = true;

      // RTM Message Listeners
      this.rtmClient.on("MessageFromPeer", (message, peerId) => {
        console.log("MessageFromPeer");
        console.log("message: ", message);
        console.log("peerId: ", peerId);
      });

      // Display connection state changes
      this.rtmClient.on("ConnectionStateChanged", (state, reason) => {
        console.log("ConnectionStateChanged");
        console.log("state: ", state);
        console.log("reason: ", reason);
      });
      // Emitted when a Call Invitation is sent from Remote User
      this.rtmClient.on("RemoteInvitationReceived", (data) => {
        this.remoteInvitation = data;
        this.incomingCall = true;
        this.incomingCaller = data.callerId;
        this.incomingCallNotification = `Incoming Call From ${data.callerId}`;

        data.on("RemoteInvitationCanceled", () => {
          console.log("RemoteInvitationCanceled: ");
          this.incomingCallNotification = "Call has been cancelled";
          setTimeout(() => {
            this.incomingCall = false;
          }, 5000);
        });
        data.on("RemoteInvitationAccepted", (data) => {
          console.log("REMOTE INVITATION ACCEPTED: ", data);
        });
        data.on("RemoteInvitationRefused", (data) => {
          console.log("REMOTE INVITATION REFUSED: ", data);
        });
        data.on("RemoteInvitationFailure", (data) => {
          console.log("REMOTE INVITATION FAILURE: ", data);
        });
      });

      // Subscribes to the online statuses of all users apart from
      // the currently authenticated user
      this.rtmClient.subscribePeersOnlineStatus(
        this.users
          .map((user) => user.username)
          .filter((user) => user !== AUTH_USER)
      );

      this.rtmClient.on("PeersOnlineStatusChanged", (data) => {
        this.updatedOnlineStatus = data;
      });

      // Create a channel and listen to messages
      this.rtmChannel = this.rtmClient.createChannel(this.channelName);

      // Join the RTM Channel
      this.rtmChannel.join();

      this.rtmChannel.on("ChannelMessage", (message, memberId) => {
        console.log("ChannelMessage");
        console.log("message: ", message);
        console.log("memberId: ", memberId);
      });

      this.rtmChannel.on("MemberJoined", (memberId) => {
        console.log("MemberJoined");

        // check whether user exists before you add them to the online user list
        const joiningUserIndex = this.onlineUsers.findIndex(
          (member) => member === memberId
        );
        if (joiningUserIndex < 0) {
          this.onlineUsers.push(memberId);
        }
      });

      this.rtmChannel.on("MemberLeft", (memberId) => {
        console.log("MemberLeft");
        console.log("memberId: ", memberId);
        const leavingUserIndex = this.onlineUsers.findIndex(
          (member) => member === memberId
        );
        this.onlineUsers.splice(leavingUserIndex, 1);
      });

      this.rtmChannel.on("MemberCountUpdated", (data) => {
        console.log("MemberCountUpdated");
      });
    },

    async placeCall(calleeName) {
      // Get the online status of the user.
      // For our use case, if the user is not online we cannot place a call.
      // We send a notification to the caller accordingly.
      this.isCallingUser = true;

      this.callingUserNotification = `Calling ${calleeName}...`;
      const onlineStatus = await this.rtmClient.queryPeersOnlineStatus([
        calleeName,
      ]);

      if (!onlineStatus[calleeName]) {
        setTimeout(() => {
          this.callingUserNotification = `${calleeName} could not be reached`;

          setTimeout(() => {
            this.isCallingUser = false;
          }, 5000);
        }, 5000);
      } else {
        // Create a channel/room name for the video call
        const videoChannelName = `${AUTH_USER}_${calleeName}`;
        // Create LocalInvitation
        this.localInvitation = this.rtmClient.createLocalInvitation(calleeName);

        this.localInvitation.on(
          "LocalInvitationAccepted",
          async (invitationData) => {
            console.log("LOCAL INVITATION ACCEPTED: ", invitationData);

            // Generate an RTM token using the channel/room name
            const { data } = await this.generateToken(videoChannelName);
            this.rtcToken = data.token;
            // Initialize the agora RTM Client
            this.initializeAgora();
            // Join a room using the channel name. The callee will also join the room then accept the call
            await this.joinRoom(AGORA_APP_ID, data.token, videoChannelName);
            this.isCallingUser = false;
            this.callingUserNotification = "";
          }
        );

        this.localInvitation.on("LocalInvitationCanceled", (data) => {
          console.log("LOCAL INVITATION CANCELED: ", data);
          this.callingUserNotification = `${calleeName} cancelled the call`;
          setTimeout(() => {
            this.isCallingUser = false;
          }, 5000);
        });
        this.localInvitation.on("LocalInvitationRefused", (data) => {
          console.log("LOCAL INVITATION REFUSED: ", data);
          this.callingUserNotification = `${calleeName} refused the call`;
          setTimeout(() => {
            this.isCallingUser = false;
          }, 5000);
        });

        this.localInvitation.on("LocalInvitationReceivedByPeer", (data) => {
          console.log("LOCAL INVITATION RECEIVED BY PEER: ", data);
        });

        this.localInvitation.on("LocalInvitationFailure", (data) => {
          console.log("LOCAL INVITATION FAILURE: ", data);
          this.callingUserNotification = "Call failed. Try Again";
        });

        // set the channelId
        this.localInvitation.channelId = videoChannelName;

        // Send call invitation
        this.localInvitation.send();
      }
    },

    async cancelCall() {
      await this.localInvitation.cancel();
      this.isCallingUser = false;
    },

    async acceptCall() {
      // Generate RTC token using the channelId of the caller
      const { data } = await this.generateToken(
        this.remoteInvitation.channelId
      );

      this.rtcToken = data.token;

      // Initialize AgoraRTC Client
      this.initializeAgora();

      // Join the room created by the caller
      await this.joinRoom(
        AGORA_APP_ID,
        data.token,
        this.remoteInvitation.channelId
      );

      // Accept Call Invitation
      this.remoteInvitation.accept();
      this.incomingCall = false;
      this.callPlaced = true;
    },

    declineCall() {
      this.remoteInvitation.refuse();
      this.incomingCall = false;
    },

    async generateToken(channelName) {
      return await axios.post(
        "/agora-cloud-recording/token",
        {
          channelName,
        },
        {
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": CSRF_TOKEN,
          },
        }
      );
    },

    /**
     * Agora Events and Listeners
     */
    initializeAgora() {
      this.rtcClient = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });
    },

    async joinRoom(appID, token, channel) {
      try {
        await this.rtcClient.join(appID, channel, token, AUTH_USER_ID);
        this.callPlaced = true;
        this.createLocalStream();
        this.initializedAgoraListeners();
      } catch (error) {
        console.log(error);
      }
    },

    initializedAgoraListeners() {
      //   Register event listeners
      this.rtcClient.on("user-published", async (user, mediaType) => {
        await this.rtcClient.subscribe(user, mediaType);

        // If the remote user publishes a video track.
        if (mediaType === "video") {
          // Get the RemoteVideoTrack object in the AgoraRTCRemoteUser object.
          this.remoteVideoTrack = user.videoTrack;
          this.remoteVideoTrack.play("remote-video");
        }
        // If the remote user publishes an audio track.
        if (mediaType === "audio") {
          // Get the RemoteAudioTrack object in the AgoraRTCRemoteUser object.term
          this.remoteAudioTrack = user.audioTrack;
          // Play the remote audio track. No need to pass any DOM element.
          this.remoteAudioTrack.play();
        }
      });

      this.rtcClient.on("user-unpublished", (data) => {
        console.log("USER UNPUBLISHED: ", data);
        // await this.endCall();
      });
    },

    async createLocalStream() {
      const [microphoneTrack, cameraTrack] =
        await AgoraRTC.createMicrophoneAndCameraTracks();
      await this.rtcClient.publish([microphoneTrack, cameraTrack]);
      cameraTrack.play("local-video");
      this.localAudioTrack = microphoneTrack;
      this.localVideoTrack = cameraTrack;
    },

    async endCall() {
      this.localAudioTrack.close();
      this.localVideoTrack.close();
      this.localAudioTrack.removeAllListeners();
      this.localVideoTrack.removeAllListeners();
      await this.rtcClient.unpublish();
      await this.rtcClient.leave();
      this.callPlaced = false;
    },

    async handleAudioToggle() {
      if (this.mutedAudio) {
        await this.localAudioTrack.setMuted(!this.mutedAudio);
        this.mutedAudio = false;
      } else {
        await this.localAudioTrack.setMuted(!this.mutedAudio);
        this.mutedAudio = true;
      }
    },

    async handleVideoToggle() {
      if (this.mutedVideo) {
        await this.localVideoTrack.setMuted(!this.mutedVideo);
        this.mutedVideo = false;
      } else {
        await this.localVideoTrack.setMuted(!this.mutedVideo);
        this.mutedVideo = true;
      }
    },

    async getResourceId() {
      const channelName =
        this.localInvitation?.channelId || this.remoteInvitation?.channelId;
      try {
        const res = await axios.post(
          "/agora-cloud-recording/resource-id",
          {
            channelName,
          },
          {
            headers: {
              "Content-Type": "application/json;charset=utf-8",
              "X-CSRFToken": CSRF_TOKEN,
            },
          }
        );
        console.log(res);
        if (res.status === 200) {
          return res.data.resourceId;
        }
        throw new Error("An error occured");
      } catch (error) {
        console.log(error);
      }
    },

    async startRecording() {
      const resourceId = await this.getResourceId();
      const channelName =
        this.localInvitation?.channelId || this.remoteInvitation?.channelId;
      try {
        const res = await axios.post(
          "/agora-cloud-recording/start-recording",
          {
            channelName,
            resourceId,
            token: this.rtcToken,
          },
          {
            headers: {
              "Content-Type": "application/json;charset=utf-8",
              "X-CSRFToken": CSRF_TOKEN,
            },
          }
        );

        if (res.status === 200) {
          this.resourceId = res.data.resourceId;
          this.recordingSID = res.data.sid;
          this.isRecordingStarted = true;
        }

        return res.data;
      } catch (error) {
        console.log(error);
      }
    },

    async stopRecording() {
      const channelName =
        this.localInvitation?.channelId || this.remoteInvitation?.channelId;
      try {
        const res = await axios.post(
          "/agora-cloud-recording/stop-recording",
          {
            channelName,
            resourceId: this.resourceId,
            token: this.rtcToken,
            sid: this.recordingSID,
          },
          {
            headers: {
              "Content-Type": "application/json;charset=utf-8",
              "X-CSRFToken": CSRF_TOKEN,
            },
          }
        );
        if (res.status === 200) {
          this.isRecordingStarted = false;
        }
        console.log(res);
      } catch (error) {
        console.log(error);
      }
    },
  },
});
