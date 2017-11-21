using UnityEngine;
using System.Collections;
using System.Linq;
using System.Collections.Generic;

//wenn man das spiel komplett neu starten können wollte, muss man dafür ResetSessionTiming vom Timinscript callen

public class GameScript : MonoBehaviour {

	public string[] mode = new string[1]{"menu"};
	public GameObject OverviewCamera;
	public GameObject CarCamera;
	public GameObject MiniMapCamera;
	public GameObject MiniMapCam2;
    public CarController Car;
	public TimingScript Timing;
	public Recorder Rec;
	public UIScript UserInterface;
	public AiInterface AiInt;

	public List<string> FreezeReasons;
	public string shouldQuickpauseReason = "";
	public string shouldUnQuickpauseReason = ""; //they would be bools, but as I also need to specify the reason, they're strings ("" or content)
	public bool ShowQuickPauseGUI = false;


	// Use this for initialization
	void Start ()
	{
		ChangeDeltaTime ();
		mode = new string[1]{"menu"};
		CarCamera.SetActive(false);
		MiniMapCamera.SetActive(false);
		MiniMapCam2.SetActive (false); 
		if (!Consts.secondcamera) { 
			((Camera)MiniMapCam2.GetComponent<Camera>()).enabled = false;	
			((Camera)MiniMapCamera.GetComponent<Camera>()).orthographicSize = 57;
			MiniMapCamera.transform.localPosition= new Vector3 (MiniMapCamera.transform.localPosition.x, MiniMapCamera.transform.localPosition.y, 50);
		} else {
			((Camera)MiniMapCamera.GetComponent<Camera>()).orthographicSize = 75;
			MiniMapCamera.transform.localPosition= new Vector3 (MiniMapCamera.transform.localPosition.x, MiniMapCamera.transform.localPosition.y, 75); // Leon's preferred setting: 75
			((Camera)MiniMapCam2.GetComponent<Camera>()).orthographicSize = 15;
			MiniMapCam2.transform.localPosition= new Vector3 (MiniMapCam2.transform.localPosition.x, MiniMapCam2.transform.localPosition.y, 15); // Leon's preferred setting: 15
		}

        OverviewCamera.SetActive(true);
		if (Rec.LoadLap("fastlap"))
		{
			Timing.fastLapSet = true;
			Timing.fastestLapTime = Rec.fastestLap[Rec.fastestLap.Count-1].time;
		}
	}

	// Update is called once per frame
	void Update ()
	{
		if (shouldQuickpauseReason != "") { //since some QuickPause-functions are not threadsafe, I can only set a variable from them such that the main-thread does that instead.
			QuickPause (shouldQuickpauseReason);
			shouldQuickpauseReason = "";
		}
		if (shouldUnQuickpauseReason != "") {
			UnQuickPause (shouldUnQuickpauseReason);
			shouldUnQuickpauseReason = "";
		}

		if (Input.GetKeyDown(KeyCode.Q)) {   
			if (mode.Contains ("driving")) {
				QuickPause ("Manual");
			} else {
				UnQuickPause ("Manual"); 
			}
		}	

		// pause & exit to menu
		if (Input.GetKeyDown(KeyCode.Escape)) { SwitchMode("menu"); }
	}

	//the Quick-pause function, triggered by Q or the python-client
	public void QuickPause(string reason) {
		if (mode.Contains ("driving")) {
			FreezeReasons.Add (reason);
			UnityEngine.Debug.Log ("Freezing because " + reason);
			mode = mode.Where (val => val != "driving").ToArray ();
			UserInterface.DriveModeDisplay.text = "GAME ON PAUSE";
			Time.timeScale = 0; //only affects fixedupdate, NOT Update!!!
			CarCamera.SetActive (false); 
			ShowQuickPauseGUI = true;
		}
	}

	public void UnQuickPause(string reason) {
		if ( (!mode.Contains ("driving")) && (!mode.Contains ("menu"))) {
			if (reason == "All")
				FreezeReasons.Clear();
			else
				FreezeReasons.Remove (reason);
			if (!FreezeReasons.Any()) {
				if (AiInt.AIMode || mode.Contains("train_AI")) {
					AiInt.load_infos(true, false);
					AiInt.lastgetvectortime = AiInterface.MSTime ();
				}
				mode = (mode ?? Enumerable.Empty<string> ()).Concat (new[] { "driving" }).ToArray ();
				UserInterface.UpdateGameModeDisp ();
				CarCamera.SetActive (true);
				Time.timeScale = 1; //TODO: auf den alten wert, wenn ich variable zeit erlaube 
				if (AiInt.AIMode)
					AiInt.load_infos (false, true);
			}
		}
	}



	void ChangeDeltaTime() {
		Time.fixedDeltaTime = 0.005f;
//		float tmpval = Time.timeScale;
//		if (SystemInfo.graphicsMemorySize > 2000)
//			tmpval = tmpval/2.0f;
//		if (tmpval >= 3)
//			Time.fixedDeltaTime = 0.01f;
//		else if (tmpval >= 1.5)
//			Time.fixedDeltaTime = 0.005f;
//		else 
//			Time.fixedDeltaTime = 0.002f;
	}

	// handle changes to the game mode
	public void SwitchMode (string newMode)
	{
		//TODO! wenn man escape drückt soll der ein ANDERES menü öffnen "gehe zu letztem checkpoint, resette auto, zurück" (da drin wäre dann auch nicht timing.reset sondern timing.stop)
		//TODO: dann beim resetten auf die resets in carcontroller, recorder, timingscript, ... achten!

		UnQuickPause ("All");
		if (AiInt.AIMode) {
			AiInt.SenderClient.ResetServerConnectTrials ();
			AiInterface.KillOtherThreads ();
			AiInt.SenderClient.serverdown = true;
		}

		if (newMode == "driving") {
			mode = new string[2]{ "driving", "keyboarddriving" };  
		} else

		if (newMode == "train_AI") {
			mode = new string[3]{ "driving", "train_AI", "keyboarddriving" };
		} else if (newMode == "drive_AI") {
			mode = new string[2]{ "driving", "drive_AI" };
		} else 	{
			mode = new string[1]{newMode};
		}
		
		AiInt.AIMode = false;
		Rec.SV_SaveMode = false;
		CarCamera.SetActive (false);
		MiniMapCamera.SetActive (false);
		if (Consts.secondcamera) { MiniMapCam2.SetActive (false); }
		OverviewCamera.SetActive (false);
		Car.ResetCar (true);
		Timing.ResetTiming ();
		UserInterface.UpdateGameModeDisp ();
		//Rec.ResetLap() ist überflüssig


		if (mode.Contains("menu")) { //geht er hin wenn man escape drückt
			OverviewCamera.SetActive (true);
		} 

		if (mode.Contains("driving")) {
			CarCamera.SetActive (true);
			MiniMapCamera.SetActive (true);
			if (Consts.secondcamera) { MiniMapCam2.SetActive (true); }
		}

		if (mode.Contains("train_AI")) {
			Rec.StartedSV_SaveMode();
		} 

		if (mode.Contains("drive_AI")) {
			AiInt.StartedAIMode ();
			AiInt.SenderClient.serverdown = false; 
		} 
			

	}

	public string UpdateGameModeDisplay() {
		string modetxt = "";
		foreach (string curr in mode) {
			modetxt += curr + " ";
		}	
		if (AiInt.HumanTakingControl) {
			modetxt += "(Human interference)";
		}
		return modetxt;
	}


}
