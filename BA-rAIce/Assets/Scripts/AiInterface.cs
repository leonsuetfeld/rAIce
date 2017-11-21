using UnityEngine;
using System.Collections;
using System;
using System.Threading;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

public static class Consts { //TODO: diese hier an python schicken!
	public const bool DEBUG_DISABLEGUI_AIMODE = true;    //a bug in Unity causes it to crash if GUI elements are updated to often... 
	public const bool DEBUG_DISABLEGUI_HUMANMODE = false; //according to my ticket https://fogbugz.unity3d.com/default.asp?935432_h1bir10rkmbc658k, this is all fixed very soon

	public const int PORTSEND = 6435;
	public const int PORTASK = 6436;
	public const int updatepythonintervalms = 100;  //multiples of 25
	public const int trackAllXMS = 25;             //hier gehts ums sv-tracken (im recorder) 
	public const int MAX_PYTHON_RT = 50;
	public const bool fixedresultusagetime = true;
	public const bool interpolateresults = false;
	public const int maxSpeed = 9999; //!!! wenn das aktiv ist KANN er nicht shcneller als maxSpeed fahren. zum testen, das macht das rennen einfacher.

	public const bool debug_showperpendicular = false;
	public const bool debug_showanchors = false;
	public const bool debug_makevalidafterwallhit = false;
	public const bool debug_updateonlyifnew = false;

	public const bool sei_verzeihender = false;
	public const bool wallhit_means_reset = true;

	public const bool usecameras = true; //only for AI-mode
	public const bool secondcamera = true; //the sizes of the cameras are set in the Start() of GameScript //if you disable this, it will be completely disabled!
	public const bool SeeCurbAsOff = false;
	public const int visiondisplay_x = 30; //30
	public const int visiondisplay_y = 45; //45
	public const int visiondisp2_x = 30; //30
	public const int visiondisp2_y = 45; //45

	public const bool trainAIMode_RestartAfterRound = true; //im drive_AI-modus entscheidet python ob am ende einer runde resettet wird, im train_AI-modus entscheidet das diese Variable.
	public const bool trainAIMode_RestartAfterWallhit = true;
}

//================================================================================

public class AiInterface : MonoBehaviour {

	public WheelCollider colliderRL;
	public WheelCollider colliderRR;
	public WheelCollider colliderFL;
	public WheelCollider colliderFR;

	public PositionTracking Tracking;
	public CarController Car;
	public MinimapScript Minmap;
	public MinimapScript Minmap2;
	public GameScript Game;
	public Recorder Rec;

	// for debugging
	public GameObject posMarker1;
	public GameObject posMarker2;
	public GameObject posMarker3;
	public GameObject posMarker4;

	//for sending to python
	public long lastpythonupdate;
	public long penultimatepythonupdate; //für das auflösen des freezes im reciever, weil python manchmal das letzte nicht zurücksendet... -.-
	public long lastgetvectortime;
	public long lastresultusagetime;
	public long lastsaviortime;
	public string lastpythonsent;
	public AsynchronousClient.Response lastUsedPythonResult;
	public FixedSizedQueue<AsynchronousClient.Response> lastpythonresults;
	public Vector3 lastCarPos;
	public Quaternion lastCarRot;
	public String lastCarAct;
	public LayerMask RayCastMask;

	//these are only for plotting and seeing if everything is ok
	public int lastunityinbetweentime;

	public float nn_steer = 0;
	public float nn_brake = 0;
	public float nn_throttle = 0;
	public bool HumanTakingControl = false;
	public bool just_hit_wall = false;
	public bool AIMode = false;
	public bool forbidSpeed = false;

	public AsynchronousClient SenderClient;
	public AsynchronousClient ReceiverClient; 

	//=============================================================================



	void Start () {
		if (Application.unityVersion != "2017.1.0f3") 
			UnityEngine.Debug.Log ("As you're using another version of Unity than the one tested with, you can maybe turn on the GUI elements savely! (variable DEBUG_DISABLEGUI in AIInterface)");
		StartedAIMode ();
	}


	public void StartedAIMode() {
		if (Game.mode.Contains ("drive_AI")) {  
			UnityEngine.Debug.Log ("Started AI Mode");
			SenderClient   = new AsynchronousClient(true, Car, this);
			ReceiverClient = new AsynchronousClient(false, Car, this);
			SendToPython ("resetServer");
			ConnectAsReceiver ();
			lastCarPos = Car.Car.position;
			lastCarRot = Car.Car.rotation;
			resetTimes ();
			lastpythonsent = "";
			lastUsedPythonResult = new AsynchronousClient.Response (null, null);
			lastpythonresults = new FixedSizedQueue<AsynchronousClient.Response> (20);
			AIMode = true;
			if (Consts.usecameras) {
				Minmap.PrepareVision (Consts.visiondisplay_x, Consts.visiondisplay_y);
				Minmap2.PrepareVision (Consts.visiondisp2_x, Consts.visiondisp2_y);
			}
		}
	}


	public void resetTimes() {
		lastpythonupdate =  UnityTime();
		lastgetvectortime = UnityTime();	
		lastresultusagetime = UnityTime();	
		Rec.lasttrack = UnityTime ();
	}


	// Update is called once per frame, and, in contrast to FixedUpdate, also runs when the game is frozen, hence the UnQuickPause here
	void Update () {
		long currtime = MSTime ();
		if (AIMode) {
			if (ReceiverClient.response.othercommand && ReceiverClient.response.command == "pleaseUnFreeze") //this must be in Update, because if the game is frozen, FixedUpdate won't run.
				Game.UnQuickPause ("Python");

			if (currtime - lastsaviortime > 2000) { //manchmal unfreezed er nicht >.< also alle 10 sekunden einfach mal machen, hässlicher workaround I know
				Game.UnQuickPause ("ConnectionDelay");
				lastsaviortime = currtime;
			}

			//RECEIVING the special commands from python
			//handle special commands...
			if (ReceiverClient.response.othercommand) {
				if (ReceiverClient.response.command == "pleasereset") { 
					Car.ResetCar (false); //false weil, wenn python dir gesagt hast dass du dich resetten sollst, du nicht python das noch sagen sollst
				}
				if (ReceiverClient.response.command == "pleaseFreeze") { 
					Game.QuickPause ("Python");
				}
				if (ReceiverClient.response.command == "pleaseUnFreeze") { 
					Game.UnQuickPause ("Python"); //is useless here, because FixedUpdate is not run during Freeze
				} 
				ReceiverClient.response.othercommand = false;
			}
		}

		// reconnect to server
		if (Input.GetKeyDown(KeyCode.C)) {   
			if (AIMode) {
				Reconnect(); 
			}
		}

		// disconnect from server
		if (Input.GetKeyDown (KeyCode.D)) { 
			if (AIMode) {
				Disconnect(); 
			}
		}		

		//human taking control over AI
		if (Input.GetKeyDown (KeyCode.H)) { 
			if (AIMode && (!Game.mode.Contains ("keyboarddriving"))) {
				FlipHumanTakingControl();
				Game.UserInterface.UpdateGameModeDisp ();
			}
		}
	}


	// FixedUpdate is run every physics-timestep, which is independent of framerate and times precisely in Unity-time
	void FixedUpdate() {
		long currtime = MSTime ();
		long currUnTime = UnityTime ();

		//SENDING data to python (all updatepythonintervalms ms)
		if (AIMode) {
			LoadAndSendToPython (false);
		}


		//RECEIVING the result from python
		if (AIMode && !HumanTakingControl) {

			// reset based on standstill
			if (long_standstill()) {
				Car.ResetCar (true);
				UnityEngine.Debug.Log("Car reset due to standstill > 2.0s");
			}
			
			//only the commands on how to drive, as special commands need to also run when frozen (hence in update)
			string message = ReceiverClient.response.getContent();
			if (message.Length > 5) {
				if (Consts.fixedresultusagetime) {
					if (!ReceiverClient.response.used)
						//print (ReceiverClient.response.getContent()); //(ohne HumanTakingControl) sieht man hier manchmal dass //python ihn los fahren lassen möchte er aber nicht will..!
						lastpythonresults.Enqueue (ReceiverClient.response.Clone());
				} else {
					float[] controls = Array.ConvertAll (message.Split (','), float.Parse);
					nn_throttle = controls [0];
					nn_brake = controls [1];
					nn_steer = controls [2];
				}
			}

 			//if you just added them in the upper part, here is where you take the fitting one (because fixed time & interpolating in between) (note that for this, unitytime is the relevant one)
			if (Consts.fixedresultusagetime) {
				if (Consts.interpolateresults) {
					if (currUnTime - lastpythonresults.Peek().CTimestampStarted >= Consts.MAX_PYTHON_RT) {
						//if you dont need lastUsedPythonResult then get rid of it
					}

				} else {
					if (lastpythonresults.Peek() != null && lastpythonresults.Peek().getContent ().Length > 1) { //if there are new items in the buffer
						if (lastpythonresults.Peek ().CTimestampStarted <= currUnTime - Consts.MAX_PYTHON_RT) { //if enough time passed that there is a new candidate..
							while (lastpythonresults.Peek() != null && lastpythonresults.Peek ().CTimestampStarted <= currUnTime - 2 * Consts.MAX_PYTHON_RT)  //get rid of the ones that are too old 
								lastpythonresults.Dequeue ();
							if (lastpythonresults.Peek () != null && lastpythonresults.Peek ().getContent ().Length > 1) {//look if there's a new one that fits
								//print ("It is: "+currUnTime + " using the result from "+(currUnTime - lastpythonresults.Peek ().CTimestampStarted)+ "ms ago. (" + lastpythonresults.length + "items in Buffer)");
								float[] controls = Array.ConvertAll (lastpythonresults.Dequeue ().getContent ().Split (','), float.Parse); //use it, and remove it from the queue!
								nn_throttle = controls [0];
								nn_brake = controls [1];
								nn_steer = controls [2];
							}
						}
					} //else do nothing, you rather need to wait
				}
//				bool dontinterpolate = false;
//				if (Consts.interpolateresults) {
//					if (lastpythonresult.getContent().Length > 1 && penultimatepythonresult.getContent().Length > 1) {
//						float[] oldcontrols = Array.ConvertAll (penultimatepythonresult.getContent().Split (','), float.Parse);
//						float[] newcontrols = Array.ConvertAll (lastpythonresult.getContent().Split (','), float.Parse);
//						float percentage = ((float)currtime - (float)lastresultusagetime) / (float)Consts.updatepythonintervalms;
//						nn_throttle = (float)((1.0 - percentage) * oldcontrols [0] + percentage * newcontrols [0]);
//						nn_brake = (float)((1.0 - percentage) * oldcontrols [1] + percentage * newcontrols [1]); 
//						nn_steer = (float)((1.0 - percentage) * oldcontrols [2] + percentage * newcontrols [2]); 
//					} else
//						dontinterpolate = true;
//				}
//				if (currUnTime - lastresultusagetime >= Consts.updatepythonintervalms-2) {
//					if (!Consts.interpolateresults || dontinterpolate) {					
//						if (lastpythonresult.getContent().Length > 1) {
//							print ((currUnTime - lastpythonresult.CTimestampStarted));
//							float[] controls = Array.ConvertAll (lastpythonresult.getContent().Split (','), float.Parse);
//							nn_throttle = controls [0];
//							nn_brake = controls [1];
//							nn_steer = controls [2];
//						}
//					}
//					lastresultusagetime = lastresultusagetime + (long)Consts.updatepythonintervalms;
//				}
			}
			nn_brake = nn_brake <= 1 ? nn_brake : 1; nn_brake = nn_brake >= 0 ? nn_brake : 0;
			nn_throttle = nn_throttle <= 1 ? nn_throttle : 1; nn_throttle = nn_throttle >= 0 ? nn_throttle : 0;
		}
	}


	public void FlipHumanTakingControl(bool force_overwrite = false, bool overwrite_with = false) {
		if (!force_overwrite) {
			HumanTakingControl = !HumanTakingControl;
		} else {
			HumanTakingControl = overwrite_with;
		}
		ReceiverClient.response.reset ();
	}
		

	public void resetCarAI() {
		if (AIMode) {
			ReceiverClient.response.reset ();
			nn_brake = 0;
			nn_steer = 0;
			nn_throttle = 0;
		}
	}


	public bool notify_wallhit() {
		if (AIMode) {
			SendToPython ("wallhit"); //ist das doppelt gemoppelt?
			just_hit_wall = true;
			return true;
		} 
		return false;
	}

	public void EndRound(bool lapClean) {
		if (AIMode) {
			SendToPython("endround"+(lapClean?"1":"0"));
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////// Sending to python //////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

	public string load_infos(Boolean force_reload, Boolean forbid_reload) {
		long currUnTime = UnityTime (); 
		bool reload = true;
        //hier waren originally sachen die verhindern sollten dass wenn drive_AI und train_AI-Modus gleichzeitig laufen die FUnktion zu oft aufgerufen wird sondern ggf das letzte result returned
	    //wenn das laden kurz genug her ist... turns out das ist schlechter so.
		//erstens weil das eh FPS-mäßig nix ändert, auch bei 40FPS, und zweitens weil dann die Loadallinfos nicht *genau* so lange her sind wie sie sollen, sondern >= so lange wie sie sollen, which is wrong.
		if (Consts.debug_updateonlyifnew) {
			Vector3 pos = Car.Car.position;
			Quaternion rot = Car.Car.rotation;
			String act = ("" + Math.Round (Car.throttlePedalValue, 3) + "," + Math.Round (Car.brakePedalValue, 3) + "," + Math.Round (Car.steeringValue, 3));

			if (pos != lastCarPos || rot != lastCarRot || act != lastCarAct) {
				lastCarPos = pos;
				lastCarRot = rot;
				lastCarAct = act;
			} else {
				reload = false;
			}
		}
		if (Input.GetKey (KeyCode.Alpha0) || Input.GetKey (KeyCode.Alpha1) || Input.GetKey (KeyCode.Alpha2) || Input.GetKey (KeyCode.Alpha3) || Input.GetKey (KeyCode.Alpha4) ||
		      Input.GetKey (KeyCode.Alpha5) || Input.GetKey (KeyCode.Alpha6) || Input.GetKey (KeyCode.Alpha7) || Input.GetKey (KeyCode.Alpha8) || Input.GetKey (KeyCode.Alpha9))
			reload = true;
		if (forbid_reload)
			reload = false;
		if (force_reload)
			reload = true;
		if (lastpythonsent == "")
			reload = true;
		
		if (reload) {
			lastpythonsent = GetAllInfos ();
			lastgetvectortime = currUnTime;
		}

		return lastpythonsent;
	}


	public void LoadAndSendToPython(Boolean force) {
		if (!AIMode) {return;}
		long currUnTime = UnityTime ();

		if (((currUnTime - lastpythonupdate >= Consts.updatepythonintervalms)) || (force)) { //FOR SENDING THE UNITYTIME IS RELEVANT; FOR MEASURING HOW LONG PYTHON TOOK THE REAL_TIME IS RELEVANT
			just_hit_wall = false;
			penultimatepythonupdate = lastpythonupdate;
			lastunityinbetweentime = (int)(currUnTime - lastpythonupdate);
			lastpythonupdate = lastpythonupdate + Consts.updatepythonintervalms; //würde nicht so klappen wenn ich realtime nehmen würde
			//print("SENDING REALTIME: " + currtime + "(" + (currUnTime-lastpythonupdate) + "ums after last)");
			SendToPython(load_infos(false, false));
		}
	}



	public void SendToPython(string data) {
		if (!AIMode) {return;}

		long currtime = MSTime ();
		if (data == "resetServer") {
			SenderClient.StartClientSocket ();
			data += Consts.updatepythonintervalms; //hier weist er python auf die fps hin
		} else {
			data = "STime(" + currtime + ")" + data;
		}

		var t =	 new Thread(() => SenderClient.SendInAnyCase(data));
		t.Start();

	}


	public void ConnectAsReceiver() {
		ReceiverClient.serverdown = false;
		ReceiverClient.StartClientSocket ();

		var t = new Thread(() => ReceiverClient.StartReceiveLoop());
		t.Start();
	}



	public void Reconnect() {
		if (!AIMode) {return;}
		Disconnect ();
		UnityEngine.Debug.Log ("Connecting...");
		SenderClient.serverdown = false;
		SenderClient.ResetServerConnectTrials();
		ReceiverClient.serverdown = false;
		ReceiverClient.ResetServerConnectTrials();
		ConnectAsReceiver ();
		SendToPython ("resetServer");
	}


	public void Disconnect() {
		if (!AIMode) {return;}
		if (!SenderClient.serverdown) {
			UnityEngine.Debug.Log ("Disconnecting...");
			SenderClient.StopClient ();
			SenderClient.serverdown = true;
			ReceiverClient.StopClient ();
			ReceiverClient.serverdown = true;
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////// Main Getter-functions ////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 


	public string GetAllInfos() {
		//Keys: P: Progress as a real number in percent, Laptime rounded to 2, lapcount, validlap
		//      S: SpeedSteerVec (rounded to 4)
		//		T: CarStatusVec  (rounded to 4)
		//		C: CenterDistVec (rounded to 4)
		//		W: Wall-Distance (7 values, from different positions with different directions)
		//		L: LookAheadVec  (rounded to 4)
		//		D: Delta & Feedback
		//		A: Action in Unity (may be different than what python asked for because of humantakingcontrol)
		//	   V1: VisionVector1 (converted to decimal)
		//	   V2: VisionVector2 (converted to decimal) (if needed)
		//		R: Progress as a vector (rounded to 4) 
		//  CTime: CreationTime of Vector (Not send-time) (this one is in Unity-Time only! The STime will be in real time)

		StringBuilder all = new StringBuilder(1902);

		all.Append ("CTime("+UnityTime().ToString()+")");

		all.Append ("P("+Math.Round(Tracking.progress * 100.0f ,3).ToString ()+","+Math.Round (Car.Timing.currentLapTime, 2).ToString ()+","+Car.Timing.lapCount.ToString ()+","+Car.lapClean.ToString () [0]+")");

		all.Append ("S("+string.Join (",", GetSpeedSteer ().Select (x => (Math.Round (x, 4)).ToString ()).ToArray ())+")");

		all.Append ("T("+string.Join (",", GetCarStatusVector ().Select (x => (Math.Round (x, 4)).ToString ()).ToArray ())+")");

		float tmp = Tracking.GetCenterDist ();
		if (just_hit_wall) 
			tmp = 11; //10 ist die distanz der mauer, aber da er ja direkt resettet weiß er es anderenfalls nicht mehr
		all.Append ("C("+Math.Round(tmp,3).ToString()+","+string.Join (",", GetCenterDistVector ().Select (x => (Math.Round(x,4)).ToString ()).ToArray ())+")");

		all.Append ("W(" + string.Join (",", GetStraightWallDist ().Select (x => (Math.Round (x, 4)).ToString ()).ToArray ()) + ")");

		all.Append ("L("+ string.Join (",", GetLookAheadVector ().Select (x => (Math.Round (x, 4)).ToString ()).ToArray ())+")");

		all.Append ("D("+Math.Round (Rec.GetDelta (), 2).ToString () + "," + Math.Round (Rec.GetFeedback (), 2).ToString ()+")");

		float maybefakethrottle;
		if (Input.GetKey (KeyCode.P))
			maybefakethrottle = 1;
		else
			maybefakethrottle = Car.throttlePedalValue;

		all.Append ("A(" + Math.Round (maybefakethrottle, 3)+","+Math.Round (Car.brakePedalValue, 3)+","+Math.Round (Car.steeringValue, 3)+")");

		if (Consts.usecameras) {
			all.Append ("V1(" +  Minmap.GetVisionDisplay () + ")");
			
			if (Consts.secondcamera)
				all.Append ("V2(" + Minmap2.GetVisionDisplay () + ")");
		}
		
		//all += "R"+ string.Join (",", GetProgressVector ().Select (x => (Math.Round(x,4)).ToString ()).ToArray ()) + ")";

		//TODO: vom carstatusvektor fehlen noch ganz viele

		return all.ToString ();
	}



	public float[] GetStraightWallDist() {
		Vector3 pos = Car.Car.position;
		Vector3 pos2 = Game.Timing.Rec.Tracking.GetPerpendicular(Car.transform.position);
		float[] result = new float[7] {300, 300, 300, 300, 300, 300, 300};

		Quaternion rot = Car.Car.transform.rotation;
		rot.x = 0;
		rot.z = 0;

		RaycastHit hit;

		Ray ray = new Ray(pos, rot*Vector3.forward);
		if (Physics.Raycast (ray, out hit, 300, RayCastMask)) {
			UnityEngine.Debug.DrawLine (Car.Car.transform.position, hit.point, Color.black, 0.08f);
			result[0] = Vector3.Distance (ray.origin, hit.point);//direction the car FACES
		}

		rot.w = rot.w - 0.1f * Car.steeringValue;

		ray = new Ray(pos, rot*Vector3.forward);
		if (Physics.Raycast (ray, out hit, 300, RayCastMask)) {
			UnityEngine.Debug.DrawLine (Car.Car.transform.position, hit.point, Color.magenta, 0.08f);
			result[1] = Vector3.Distance (ray.origin, hit.point); //direction the car STEERS
		}

		rot = Quaternion.LookRotation (Car.transform.position - Car.lastb1Position);
		rot.x = 0;
		rot.z = 0;

		ray = new Ray (pos, rot*Vector3.forward);
		if (Physics.Raycast (ray, out hit, 300, RayCastMask)) {
			UnityEngine.Debug.DrawLine (Car.Car.transform.position, hit.point, Color.red, 0.08f);
			result[2] = Vector3.Distance (ray.origin, hit.point); //direction the car MOVES
		}


		//SHORTSIGHTED...
		rot = Quaternion.LookRotation (Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position) - Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position, 2));

		ray = new Ray (pos, rot*Vector3.forward);
		if (Physics.Raycast (ray, out hit, 300, RayCastMask)) {
			UnityEngine.Debug.DrawLine (Car.Car.transform.position, hit.point, Color.white, 0.08f);
			result[3] = Vector3.Distance (ray.origin, hit.point); //shortsighted from car, direction the STREET GOES
		}

		rot = Quaternion.LookRotation (Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position) - Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position, 2));

		ray = new Ray (pos2, rot*Vector3.forward);
		if (Physics.Raycast (ray, out hit, 300, RayCastMask)) {
			UnityEngine.Debug.DrawLine (Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position), hit.point, Color.yellow, 0.08f);
			result[4] = Vector3.Distance (ray.origin, hit.point); //shortsighted from STREETMIDDLE, direction the STREET GOES
		}

		//LONGSIGHTED...
		rot = Quaternion.LookRotation (Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position, -4) - Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position));

		ray = new Ray (pos, rot*Vector3.forward);
		if (Physics.Raycast (ray, out hit, 500, RayCastMask)) {
			UnityEngine.Debug.DrawLine (Car.Car.transform.position, hit.point, Color.blue, 0.08f);
			result[5] = Vector3.Distance (ray.origin, hit.point); //longsighted from car, direction the STREET GOES
		}

		rot = Quaternion.LookRotation (Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position, -4) - Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position));

		ray = new Ray (pos2, rot*Vector3.forward);
		if (Physics.Raycast (ray, out hit, 500, RayCastMask)) {
			UnityEngine.Debug.DrawLine (Game.Timing.Rec.Tracking.GetPerpendicular (Car.transform.position), hit.point, Color.grey, 0.08f);
			result[6] = Vector3.Distance (ray.origin, hit.point); //longsighted from STREETMIDDLE, direction the STREET GOES
		}

		//UnityEngine.Debug.Log (result [1] - Car.velocity + 100);
		//supergute speed-heuristik: je näher dieser wert an 0 ist, desto besser. wobei über 0 flach steigt, unter 0 steil: result [1] - Car.velocity + 100

		return result;
	}



	public float[] GetSpeedSteer() {
		float velo = Car.velocity;
		float velo2 = Car.velocityOfPerpendiculars;
		float[] velo345 = Tracking.GetSpeedInDir();
		int MAXSPEED = 250;
		int fake_speed = -1;
		if (HumanTakingControl) { //um geschwindigkeiten zu faken damit man sich die entsprechenden q-werte anschauen kann
			if (Input.GetKey (KeyCode.Alpha0)) 
				fake_speed = 0;
			if (Input.GetKey (KeyCode.Alpha1)) 
				fake_speed = 1;
			if (Input.GetKey (KeyCode.Alpha2)) 
				fake_speed = 2;
			if (Input.GetKey (KeyCode.Alpha3)) 
				fake_speed = 3;
			if (Input.GetKey (KeyCode.Alpha4)) 
				fake_speed = 4;
			if (Input.GetKey (KeyCode.Alpha5)) 
				fake_speed = 5;
			if (Input.GetKey (KeyCode.Alpha6)) 
				fake_speed = 6;
			if (Input.GetKey (KeyCode.Alpha7)) 
				fake_speed = 7;
			if (Input.GetKey (KeyCode.Alpha8)) 
				fake_speed = 8;
			if (Input.GetKey (KeyCode.Alpha9)) 
				fake_speed = 9;			
		}
		if (fake_speed > -1) {
			velo = fake_speed/9.0f * MAXSPEED;
			velo2 = fake_speed/9.0f * MAXSPEED;
			velo345[0] = fake_speed/9.0f * MAXSPEED;
			Game.UserInterface.Speedometer.text = velo.ToString() + " kph";
		}
		float RLTorque = Car.maxMotorTorque * Car.throttlePedalValue * Car.gear;
		float RRTorque = Car.maxMotorTorque * Car.throttlePedalValue * Car.gear;

		float[] SpeedSteerVec = new float[11] { RLTorque, RRTorque, colliderFL.steerAngle, colliderFR.steerAngle, velo, Convert.ToInt32(Tracking.rightDirection), velo2, Tracking.getCarAngle(), velo345[0], velo345[1], velo345[2]};
		return SpeedSteerVec;
	}



	public float[] GetCenterDistVector(int vectorLength=15, float spacing=1.0f, float sigma=1.0f) { // needs vis. display 
		// car-centered: where is the center line of the race track relative to my car?
		float centerLineDist = Tracking.GetCenterDist(); // positive if the center line is to the left, negative if it is to the right

		// RBF centers: spread out like [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
		float[] centerPositions = new float[vectorLength];
		for (int i=0; i<centerPositions.Length; i++) { centerPositions[i] = (i-((vectorLength-1)/2.0f))*spacing; }

		// centerDistVector: Gaussian distribution with centerPosition as mu, sigma as sigma, and centerLineDist as x
		float[] centerDistVector = new float[vectorLength];
		for (int j=0; j<centerDistVector.Length; j++)
		{
			float centerLineDistAdjusted = centerLineDist;
			if (j==0 && centerLineDistAdjusted < centerPositions[0]) { centerLineDistAdjusted = centerPositions[0]; }
			if (j==(centerPositions.Length-1) && centerLineDistAdjusted > centerPositions[centerPositions.Length-1]) { centerLineDistAdjusted = centerPositions[centerPositions.Length-1]; }
			centerDistVector[j] = GaussianDist(centerPositions[j], sigma, centerLineDistAdjusted);
		}
		return centerDistVector;
	}



	public float[] GetLookAheadVector(int vectorLength=30, float spacing=10.0f) {
		// track-centered: does not take into accout current rotation of the car
		float progressMeters = Tracking.progress*Tracking.trackLength;
		float[] lookAhead = new float[vectorLength];
		for (int i=0; i<vectorLength; i++)
		{
			float lookAheadPointOne = progressMeters+i*spacing;
			float lookAheadPointTwo = progressMeters+(i+1)*spacing;
			if (lookAheadPointOne > Tracking.trackLength) { lookAheadPointOne -= Tracking.trackLength; }
			if (lookAheadPointTwo > Tracking.trackLength) { lookAheadPointTwo -= Tracking.trackLength; }
			float absoluteAnglePoint1 = InterpolatePointAngle(Tracking.absoluteAnchorAngles, Tracking.absoluteAnchorDistances, lookAheadPointOne);
			float absoluteAnglePoint2 = InterpolatePointAngle(Tracking.absoluteAnchorAngles, Tracking.absoluteAnchorDistances, lookAheadPointTwo);
			float tmp = (absoluteAnglePoint2 - absoluteAnglePoint1);
			if (tmp > 180.0f) { tmp -= 360.0f; }
			if (tmp < -180.0f) { tmp += 360.0f; }
			lookAhead[i] = tmp;
		}
		return lookAhead;
	}



	public float[] GetProgressVector(int vectorLength=10, float sigma=0.1f) { // needs vis. display 
		// parameters
		float spacing = 1.0f / (float)vectorLength;

		// RBF center positions
		float[] centerPositions = new float[vectorLength];
		centerPositions[0] = 0.0f;
		for (int i=1; i<vectorLength; i++) { centerPositions[i] = centerPositions[i-1]+spacing; }

		// distances from RBF centers
		float[] progressVector = new float[vectorLength];
		for (int j=0; j<vectorLength; j++)
		{
			float distFromCenter = Mathf.Abs(centerPositions[j] - Tracking.progress);
			if (distFromCenter > 0.5f) { distFromCenter -= 1.0f; distFromCenter = Mathf.Abs(distFromCenter); }
			progressVector[j] = GaussianDist(0.0f, sigma, distFromCenter); // dist from center is put in directly, so new center is 0
		}

		return progressVector;
	}




	public float[] GetCarStatusVector() {
		float[] carStatusVector = new float[9]; // length = sum(#) ~=18
		carStatusVector[0] = Car.velocity/200.0f; // car velocity > split up into more nodes 	# 1      //why do we need both the velocity and the speed from GetSpeedSteer?
		carStatusVector[1] = Car.GetSlip(Car.colliderFL)[0]; // wheel rotation relative to car	# 1
		carStatusVector[2] = Car.GetSlip(Car.colliderFR)[0]; // wheel rotation relative to car	# 1
		carStatusVector[3] = Car.GetSlip(Car.colliderRL)[0]; // wheel rotation relative to car	# 1
		carStatusVector[4] = Car.GetSlip(Car.colliderRR)[0]; // wheel rotation relative to car  # 1
		carStatusVector[5] = Car.GetSlip(Car.colliderFL)[1]; // wheel rotation relative to car	# 1
		carStatusVector[6] = Car.GetSlip(Car.colliderFR)[1]; // wheel rotation relative to car	# 1
		carStatusVector[7] = Car.GetSlip(Car.colliderRL)[1]; // wheel rotation relative to car	# 1
		carStatusVector[8] = Car.GetSlip(Car.colliderRR)[1]; // wheel rotation relative to car  # 1
		// front wheel rotation relative to centerLine rotation									# 2
		// car rotation relative to centerLine rotation											# 2
		// car rotation relative to velocity vector												# 1
		// longitudinal slip FL																	# 1
		// longitudinal slip FR																	# 1
		// longitudinal slip RL																	# 1
		// longitudinal slip RR																	# 1
		// slip angle FL																		# 1
		// slip angle FR																		# 1
		// slip angle RL																		# 1
		// slip angle RR																		# 1
		return carStatusVector;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////// Helper-functions ///////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

	bool long_standstill()
	{
		if (UnityTime() - Car.standstill_start_time > 2000.0f) {
			return true;
		}
		return false;
	}

	float InterpolatePointAngle(float[] absoluteAnchorAngles, float[] absoluteAnchorDistances, float d)
	{
		int pos = ClosestSmallerThan(absoluteAnchorDistances, d);
		int posPlus1 = pos+1; if (posPlus1 >= absoluteAnchorDistances.Length) { posPlus1 -= absoluteAnchorDistances.Length; }
		float A = absoluteAnchorDistances[pos];				// d is either A, B, or between A and B
		float B = absoluteAnchorDistances[posPlus1];
		float interpolationRatio = (d-A)/(B-A);

		float angleA = absoluteAnchorAngles[pos];
		float angleB = absoluteAnchorAngles[posPlus1];
		if ((angleA-angleB)>180.0f) { angleA -= 360.0f; } // a=360, b=0 
		if ((angleA-angleB)<-180.0f) { angleB -= 360.0f; } // a=0, b=360 
		return (1-interpolationRatio)*angleA + interpolationRatio*angleB; // return linear interpolation between angle in point A and B
	}


	static int ClosestSmallerThan(float[] collection, float target)
	{
		float minDifference = float.MaxValue;
		int argClosest = int.MaxValue;
		for (int i=0; i<collection.Length; i++)
		{
			if (target > collection[i])
			{
				float difference = Mathf.Abs(collection[i] - target);
				if (minDifference > difference)
				{
					argClosest = i;
					minDifference = difference;
				}
			} 
		}
		if (argClosest == int.MaxValue) {
			argClosest = 0;
		}
		return argClosest;
	}

	float GaussianDist(float mu, float sigma, float x)
	{
		return 1.0f/Mathf.Sqrt(2.0f*Mathf.PI*sigma)*Mathf.Exp(-Mathf.Pow((x-mu),2.0f)/(2.0f*Mathf.Pow(sigma,2.0f)));
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////// helper-classes & functions //////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 


	public class FixedSizedQueue<T>
	{
		Queue<T> q;
		public int Limit { get; set; }
		public int length;

		public FixedSizedQueue(int limit) {
			q = new Queue<T>();
			Limit = limit;
			length = 0;
		}

		public void Enqueue(T obj)
		{
			q.Enqueue(obj);
			if (length < Limit)
				length += 1;
			
			while (q.Count > Limit)
				q.Dequeue ();
		}

		public T Peek() {
			try {
				return q.Peek ();
			} catch (InvalidOperationException E) {
				return default(T);
			}
		}

		public int getAverage() {
			List<int> list = q.Select(x => int.Parse(x.ToString())).ToList ();
			var average = list.Average();
			//print(average.ToString());
			return (int) average;
		}

		public T Dequeue() {
			if (length > 0)
				length -= 1;
			return q.Dequeue ();
		}

		public void Clear() {
			while (q.Count > 0)
				q.Dequeue ();
		}

	}

	private static int ParseIntBase3(string s)
	{
		int res = 0;
		for (int i = 0; i < s.Length; i++)
		{
			res = 3 * res + s[i] - '0';
		}
		return res;
	}


	public static void print(string str) {
		UnityEngine.Debug.Log (str);
	}

	public static long MSTime() {
		//https://stackoverflow.com/questions/243351/environment-tickcount-vs-datetime-now 
		return ((long)((long)DateTime.UtcNow.Ticks/10000L)) % 10000000000L; //there are 10000ticks in a ms
	}

	public static long UnityTime() {
		return (long)(Time.time * 1000);
	}

	public static long UTCTime() {
		//https://www.epochconverter.com/ 
		return (long)(DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
	}


	public static void KillOtherThreads() {
		ProcessThreadCollection currentThreads = Process.GetCurrentProcess().Threads;
		foreach (Thread thread in currentThreads)    
		{
			thread.Abort();
		}	
	}

}