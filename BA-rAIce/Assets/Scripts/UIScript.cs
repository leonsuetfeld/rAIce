using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Text;
using System.Linq;


//TODO: eine funktion die neu-rendern enforced (wird gecallt nach jedem reset, bspw)
//TODO: optionen: -reset all laps, -wallhit means reset, -aktiviere debugstuff  (optionen konnten auch einfach zum gamemode adden, oder halt ne eigene klasse sein)
//TODO: ein anderes pausemenu (continue, reset to last position, back to main menu)

public class UIScript : MonoBehaviour {

	public GameScript Game;

	// objects needed for driving UI
	public CarController Car;
	public TimingScript Timing;
	public PositionTracking Tracking;
	public Recorder Rec;
	public Text Speedometer;
	public Text GearDisplay;
	public Text DriveModeDisplay;
	public Text CurrentLaptime;
	public Text LastLaptime;
	public Text FastestLaptime;
	public Text Progress;
	public Text Delta;
	public Text LapCount;
	public Text FSlipFL;
	public Text FSlipFR;
	public Text FSlipRL;
	public Text FSlipRR;
	public Text SSlipFL;
	public Text SSlipFR;
	public Text SSlipRL;
	public Text SSlipRR;
	public Text FSlipHeadline;
	public Text SSlipHeadline;
	public GameObject UiBgTop;
	public GameObject UiBgBottom;
	public GameObject UiFeedbackBg;
	public Image UiFeedbackPositive;
	public Image UiFeedbackNegative;
	int speed;
	float progressValue;
	float lastUpdate = 0.0f;
	float deltaValue;
	float feedbackValue;
	private bool drivingOverlayActive = false;
	public Image throttlePedal;
	public Image brakePedal;
	public Image steeringAngle;
	public Image steeringBg;
	float steeringPos;
	public Image carPosY;
	public GameObject UiPosYBg;
	float posY;
	public AiInterface Ai;
	public MinimapScript Minmap;
	public GameObject UiLookAheadBg;
	public Image LookAheadPoint_1;
	public Image LookAheadPoint_2;
	public Image LookAheadPoint_3;
	public Image LookAheadPoint_4;
	public Image LookAheadPoint_5;
	public Image LookAheadPoint_6;
	public Image LookAheadPoint_7;
	public Image LookAheadPoint_8;
	public Image LookAheadPoint_9;
	public Image LookAheadPoint_10;
	public GameObject UiProgressVectorBg;
	public Image ProgressCell_1;
	public Image ProgressCell_2;
	public Image ProgressCell_3;
	public Image ProgressCell_4;
	public Image ProgressCell_5;
	public Image ProgressCell_6;
	public Image ProgressCell_7;
	public Image ProgressCell_8;
	public Image ProgressCell_9;
	public Image ProgressCell_10;
	public GameObject UiCenterDistanceVectorBg;
	public Image DistanceCell_1;
	public Image DistanceCell_2;
	public Image DistanceCell_3;
	public Image DistanceCell_4;
	public Image DistanceCell_5;
	public Image DistanceCell_6;
	public Image DistanceCell_7;
	public Image DistanceCell_8;
	public Image DistanceCell_9;
	public Image DistanceCell_10;

	// objects needed for menu UI
	public GameObject UiBgMenucover;
	public GameObject UiBgButtonDrive;
	public GameObject UiBgButtonTrainAI;
	public GameObject UiBgButtonDriveAI;
	public GameObject UiBgButtonOptions;
	public GameObject UiBgButtonQuit;
	public GameObject UiThrottle;
	public GameObject UiBrake;
	public GameObject UiSteering;
	public Text UiTextDrive;
	public Text UiTextTrainAI;
	public Text UiTextDriveAI;
	public Text UiTextOptions;
	public Text UiTextQuit;
	private bool menuOverlayActive = false;
	private int menuSelection = 0;

	// objects for debugging
	public Image DebugMarker;
	public Image DebugMarker2;

	// color definitions
	Color racingWhite = new Color(1.0f,1.0f,1.0f,1.0f);
	Color racingRed = new Color(1.0f,0.2f,0.2f,1.0f);
	Color racingGreen = new Color(0.3f,0.83f,0.13f,1.0f);

	// Use this for initialization
	void Start ()
	{
		GearDisplay.enabled = false;
		menuSelection = 0;
		drivingOverlayActive = true;
	}



	void OnGUI () {
		if (!((Consts.DEBUG_DISABLEGUI_AIMODE && Game.AiInt.AIMode) || Consts.DEBUG_DISABLEGUI_HUMANMODE && !Game.AiInt.AIMode)) { 
			if (Time.timeScale > 0) {
				GUI.Label (new Rect (0, 0, 100, 600), "FPS: " + ((int)(1.0f / Time.smoothDeltaTime)).ToString ());  
			} 
			if (Game.AiInt.AIMode) {
				GUI.Label (new Rect (0, 15, 100, 600), "Python RT:" + Game.AiInt.ReceiverClient.response.pythonreactiontime.ToString ());  
				GUI.Label (new Rect (0, 30, 100, 600), "i.b. sendings:" + Game.AiInt.lastunityinbetweentime.ToString ());  
			}


			if (Game.ShowQuickPauseGUI) {
				RenderTexture myRT = new RenderTexture (1, 1, 24);  //,RenderTextureFormat.ARGB32
				myRT.Create ();
				GUI.DrawTexture (new Rect (10, 10, 60, 60), myRT, ScaleMode.ScaleToFit, true, 10.0F);
			}
		}
	}


	
	// Update is called once per frame
	void Update ()
	{
		MenuOverlayHandling();
		DrivingOverlayHandling();
	}

	public void DrivingOverlayHandling()
	{
		if (Game.mode.Contains("menu") && drivingOverlayActive == true) {  
			UiBgTop.SetActive(false); UiBgBottom.SetActive(false); UiFeedbackBg.SetActive(false); UiSteering.SetActive(false); 
			UiThrottle.SetActive(false); UiBrake.SetActive(false); carPosY.enabled=false; Speedometer.enabled=false; DriveModeDisplay.enabled = false;
			FSlipFL.enabled=false; FSlipFR.enabled=false; FSlipRL.enabled=false; FSlipRR.enabled=false; SSlipFL.enabled=false; SSlipFR.enabled=false; 
			SSlipRL.enabled=false; SSlipRR.enabled=false; FSlipHeadline.enabled=false; SSlipHeadline.enabled=false; GearDisplay.enabled=false; 
			CurrentLaptime.enabled=false; Delta.enabled=false; LastLaptime.enabled=false; FastestLaptime.enabled=false; Progress.enabled=false; 
			LapCount.enabled=false; drivingOverlayActive=false; 
			UiLookAheadBg.SetActive(false); UiProgressVectorBg.SetActive(false); UiCenterDistanceVectorBg.SetActive(false); UiPosYBg.SetActive (false);

		}

		if (!((Consts.DEBUG_DISABLEGUI_AIMODE && Game.AiInt.AIMode) || Consts.DEBUG_DISABLEGUI_HUMANMODE && !Game.AiInt.AIMode)) {
			if ((Game.mode.Contains("train_AI") || Game.mode.Contains("drive_AI")) && drivingOverlayActive == false) { 
				UiLookAheadBg.SetActive(true); UiProgressVectorBg.SetActive(true); UiCenterDistanceVectorBg.SetActive(true); UiPosYBg.SetActive (true); carPosY.enabled=true; 
			}

			if (Game.mode.Contains("driving") && drivingOverlayActive == false) { 
				UiBgTop.SetActive(true); UiBgBottom.SetActive(true); UiFeedbackBg.SetActive(true); UiSteering.SetActive(true); UiThrottle.SetActive(true); UiBrake.SetActive(true); 
				Speedometer.enabled=true;  FSlipFL.enabled=true; FSlipFR.enabled=true; FSlipRL.enabled=true; FSlipRR.enabled=true; SSlipFL.enabled=true; DriveModeDisplay.enabled = true;
				SSlipFR.enabled=true; SSlipRL.enabled=true; SSlipRR.enabled=true; FSlipHeadline.enabled=true; SSlipHeadline.enabled=true; GearDisplay.enabled=true; 
				CurrentLaptime.enabled=true; Delta.enabled=true; LastLaptime.enabled=true; FastestLaptime.enabled=true; Progress.enabled=true; LapCount.enabled=true; drivingOverlayActive=true;
			}


			// update UI elements during driving
			if (Game.mode.Contains("driving"))
			{
				// gear display
				if (Car.gear < 0.0f) { GearDisplay.enabled = true; }
				else { GearDisplay.enabled = false; }
		
				// slower updates: speed, progress & wheel rotation display
				if (Time.time - lastUpdate > 0.2f)
				{
					speed = (int) Mathf.Round(Car.velocity);
					Speedometer.text = speed.ToString() + " kph";
					lastUpdate = Time.time;

					// wheel rotation relative to car
					Vector2 FL = Car.GetSlip(Car.colliderFL);
					Vector2 FR = Car.GetSlip(Car.colliderFR);
					Vector2 RL = Car.GetSlip(Car.colliderRL);		
					Vector2 RR = Car.GetSlip(Car.colliderRR);
					FSlipFL.text = FL [0].ToString ("F2");
					FSlipFR.text = FR [0].ToString ("F2");
					FSlipRL.text = RL [0].ToString ("F2");
					FSlipRR.text = RR [0].ToString ("F2");
					SSlipFL.text = FL [1].ToString ("F2");
					SSlipFR.text = FR [1].ToString ("F2");
					SSlipRL.text = RL [1].ToString ("F2");
					SSlipRR.text = RR [1].ToString ("F2");

				}

				// current laptime display
				CurrentLaptime.text = Timing.currentLapTime.ToString("F1");
				if (Timing.timeSet)
				{
					LastLaptime.text = Timing.lastLapTime.ToString("F3");
				}
				if (Timing.fastLapSet)
				{
					FastestLaptime.text = Timing.fastestLapTime.ToString("F3");
				}
		
				// current progress in per cent
				progressValue = Tracking.progress*100.0f;
				Progress.text = progressValue.ToString("F2") + "%";

				// current delta in seconds
				deltaValue = Rec.GetDelta();
				Delta.text = "(" + deltaValue.ToString("F2") + ")";
				if (deltaValue > 0.0f)
				{
					Delta.text = "(+" + deltaValue.ToString("F2") + ")";
					Delta.color = racingRed;
				}
				else if (deltaValue == 0.0f)
				{
					Delta.text = "(+" + deltaValue.ToString("F2") + ")";
					Delta.color = racingWhite;
				}
				else
				{
					Delta.text = "(" + deltaValue.ToString("F2") + ")";
					Delta.color = racingGreen;
				}

				// lap counter
				if (Car.lapClean) { LapCount.text = "lap " + Timing.lapCount.ToString(); }
				else { LapCount.text = "lap " + Timing.lapCount.ToString() + " (invalid)"; }

				// Feedback bar
				feedbackValue = Rec.GetFeedback();
				if (feedbackValue > 1.0f) { feedbackValue = 1.0f; }
				if (feedbackValue < -1.0f) { feedbackValue = -1.0f; }
				if (feedbackValue > 0.0f)
				{
					UiFeedbackPositive.rectTransform.sizeDelta = new Vector2(feedbackValue*100.0f, UiFeedbackPositive.rectTransform.sizeDelta.y);
					UiFeedbackNegative.rectTransform.sizeDelta = new Vector2(0.0f, UiFeedbackNegative.rectTransform.sizeDelta.y);
				}
				else if (feedbackValue < 0.0f)
				{
					UiFeedbackPositive.rectTransform.sizeDelta = new Vector2(0.0f, UiFeedbackPositive.rectTransform.sizeDelta.y);
					UiFeedbackNegative.rectTransform.sizeDelta = new Vector2(-feedbackValue*100.0f, UiFeedbackNegative.rectTransform.sizeDelta.y);
				}
				else if (feedbackValue == 0.0f)
				{
					UiFeedbackPositive.rectTransform.sizeDelta = new Vector2(0.0f, UiFeedbackPositive.rectTransform.sizeDelta.y);
					UiFeedbackNegative.rectTransform.sizeDelta = new Vector2(0.0f, UiFeedbackNegative.rectTransform.sizeDelta.y);
				}

				// steering, throttle & brake pedals
				steeringPos = 68.0f*Car.steeringValue; // this isn't scaling with the screensize
				steeringAngle.rectTransform.position = new Vector3(UiSteering.transform.position.x+steeringPos, steeringAngle.rectTransform.position.y, steeringAngle.rectTransform.position.z);
				throttlePedal.rectTransform.sizeDelta = new Vector2(brakePedal.rectTransform.sizeDelta.x, Car.throttlePedalValue*65.0f);
				brakePedal.rectTransform.sizeDelta = new Vector2(brakePedal.rectTransform.sizeDelta.x, Car.brakePedalValue*65.0f);


				if (Game.mode.Contains("train_AI") || Game.mode.Contains("drive_AI")) {
					// current posY
					posY = Tracking.GetCenterDist () * 7.4f;
					carPosY.rectTransform.position = new Vector3 (UiPosYBg.transform.position.x + posY, carPosY.rectTransform.position.y, carPosY.rectTransform.position.z);

					// current lookAhead
					float[] lookAheadVector = Ai.GetLookAheadVector (10, 20.0f);
					float pxOffsetLookAhead = 1.0f;
					LookAheadPoint_1.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [9] * pxOffsetLookAhead, LookAheadPoint_1.rectTransform.position.y, LookAheadPoint_1.rectTransform.position.z);
					LookAheadPoint_2.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [8] * pxOffsetLookAhead, LookAheadPoint_2.rectTransform.position.y, LookAheadPoint_2.rectTransform.position.z);
					LookAheadPoint_3.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [7] * pxOffsetLookAhead, LookAheadPoint_3.rectTransform.position.y, LookAheadPoint_3.rectTransform.position.z);
					LookAheadPoint_4.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [6] * pxOffsetLookAhead, LookAheadPoint_4.rectTransform.position.y, LookAheadPoint_4.rectTransform.position.z);
					LookAheadPoint_5.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [5] * pxOffsetLookAhead, LookAheadPoint_5.rectTransform.position.y, LookAheadPoint_5.rectTransform.position.z);
					LookAheadPoint_6.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [4] * pxOffsetLookAhead, LookAheadPoint_6.rectTransform.position.y, LookAheadPoint_6.rectTransform.position.z);
					LookAheadPoint_7.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [3] * pxOffsetLookAhead, LookAheadPoint_7.rectTransform.position.y, LookAheadPoint_7.rectTransform.position.z);
					LookAheadPoint_8.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [2] * pxOffsetLookAhead, LookAheadPoint_8.rectTransform.position.y, LookAheadPoint_8.rectTransform.position.z);
					LookAheadPoint_9.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [1] * pxOffsetLookAhead, LookAheadPoint_9.rectTransform.position.y, LookAheadPoint_9.rectTransform.position.z);
					LookAheadPoint_10.rectTransform.position = new Vector3 (UiLookAheadBg.transform.position.x + lookAheadVector [0] * pxOffsetLookAhead, LookAheadPoint_10.rectTransform.position.y, LookAheadPoint_10.rectTransform.position.z);

					// current progressVector
					float[] progressVector = Ai.GetProgressVector (10, 0.08f);
					ProgressCell_1.rectTransform.position = new Vector3 (ProgressCell_1.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [0] * 22.0f - 18.0f, ProgressCell_1.rectTransform.position.z);
					ProgressCell_2.rectTransform.position = new Vector3 (ProgressCell_2.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [1] * 22.0f - 18.0f, ProgressCell_2.rectTransform.position.z);
					ProgressCell_3.rectTransform.position = new Vector3 (ProgressCell_3.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [2] * 22.0f - 18.0f, ProgressCell_3.rectTransform.position.z);
					ProgressCell_4.rectTransform.position = new Vector3 (ProgressCell_4.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [3] * 22.0f - 18.0f, ProgressCell_4.rectTransform.position.z);
					ProgressCell_5.rectTransform.position = new Vector3 (ProgressCell_5.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [4] * 22.0f - 18.0f, ProgressCell_5.rectTransform.position.z);
					ProgressCell_6.rectTransform.position = new Vector3 (ProgressCell_6.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [5] * 22.0f - 18.0f, ProgressCell_6.rectTransform.position.z);
					ProgressCell_7.rectTransform.position = new Vector3 (ProgressCell_7.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [6] * 22.0f - 18.0f, ProgressCell_7.rectTransform.position.z);
					ProgressCell_8.rectTransform.position = new Vector3 (ProgressCell_8.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [7] * 22.0f - 18.0f, ProgressCell_8.rectTransform.position.z);
					ProgressCell_9.rectTransform.position = new Vector3 (ProgressCell_9.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [8] * 22.0f - 18.0f, ProgressCell_9.rectTransform.position.z);
					ProgressCell_10.rectTransform.position = new Vector3 (ProgressCell_10.transform.position.x, UiProgressVectorBg.transform.position.y + progressVector [9] * 22.0f - 18.0f, ProgressCell_10.rectTransform.position.z);

					// current centerDistVector
					float[] centerDistanceVector = Ai.GetCenterDistVector (10, 1.5f, 1.0f);
					DistanceCell_1.rectTransform.position = new Vector3 (DistanceCell_1.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [9] * 100.0f - 18.0f, DistanceCell_1.rectTransform.position.z);
					DistanceCell_2.rectTransform.position = new Vector3 (DistanceCell_2.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [8] * 100.0f - 18.0f, DistanceCell_2.rectTransform.position.z);
					DistanceCell_3.rectTransform.position = new Vector3 (DistanceCell_3.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [7] * 100.0f - 18.0f, DistanceCell_3.rectTransform.position.z);
					DistanceCell_4.rectTransform.position = new Vector3 (DistanceCell_4.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [6] * 100.0f - 18.0f, DistanceCell_4.rectTransform.position.z);
					DistanceCell_5.rectTransform.position = new Vector3 (DistanceCell_5.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [5] * 100.0f - 18.0f, DistanceCell_5.rectTransform.position.z);
					DistanceCell_6.rectTransform.position = new Vector3 (DistanceCell_6.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [4] * 100.0f - 18.0f, DistanceCell_6.rectTransform.position.z);
					DistanceCell_7.rectTransform.position = new Vector3 (DistanceCell_7.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [3] * 100.0f - 18.0f, DistanceCell_7.rectTransform.position.z);
					DistanceCell_8.rectTransform.position = new Vector3 (DistanceCell_8.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [2] * 100.0f - 18.0f, DistanceCell_8.rectTransform.position.z);
					DistanceCell_9.rectTransform.position = new Vector3 (DistanceCell_9.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [1] * 100.0f - 18.0f, DistanceCell_9.rectTransform.position.z);
					DistanceCell_10.rectTransform.position = new Vector3 (DistanceCell_10.transform.position.x, UiCenterDistanceVectorBg.transform.position.y + centerDistanceVector [0] * 100.0f - 18.0f, DistanceCell_10.rectTransform.position.z);
				}
			}
		}
	}

	public void UpdateGameModeDisp() {
		DriveModeDisplay.text = Game.UpdateGameModeDisplay ();
	}

	void MenuOverlayHandling()
	{
		// enable/ disable menu overlay
		if (Game.mode.Contains("menu") && menuOverlayActive == false) { UiBgMenucover.SetActive(true); UiTextDrive.enabled=true; UiTextTrainAI.enabled=true; UiTextDriveAI.enabled=true; UiTextOptions.enabled=true; UiTextQuit.enabled=true; menuOverlayActive=true; }
		if (Game.mode.Contains("driving") && menuOverlayActive == true) { UiBgMenucover.SetActive(false); UiBgButtonDrive.SetActive(false); UiBgButtonDrive.SetActive(false); UiTextDrive.enabled=false; UiTextTrainAI.enabled=false; UiTextDriveAI.enabled=false; UiTextOptions.enabled=false; UiTextQuit.enabled=false; menuOverlayActive=false; }
	
		// menu interface
		if (Game.mode.Contains("menu"))
		{
			switch (menuSelection)
			{
				case 0:
					UiBgButtonDrive.SetActive(true);
					UiBgButtonTrainAI.SetActive(false);
					UiBgButtonDriveAI.SetActive(false);
					UiBgButtonOptions.SetActive(false);
					UiBgButtonQuit.SetActive(false);
					break;
				case 1:
					UiBgButtonDrive.SetActive(false);
					UiBgButtonTrainAI.SetActive(true);
					UiBgButtonDriveAI.SetActive(false);
					UiBgButtonOptions.SetActive(false);
					UiBgButtonQuit.SetActive(false);
					break;
				case 2:
					UiBgButtonDrive.SetActive(false);
					UiBgButtonTrainAI.SetActive(false);
					UiBgButtonDriveAI.SetActive(true);
					UiBgButtonOptions.SetActive(false);
					UiBgButtonQuit.SetActive(false);
					break;
				case 3:
					UiBgButtonDrive.SetActive(false);
					UiBgButtonTrainAI.SetActive(false);
					UiBgButtonDriveAI.SetActive(false);
					UiBgButtonOptions.SetActive(true);
					UiBgButtonQuit.SetActive(false);
					break;
				case 4:
					UiBgButtonDrive.SetActive(false);
					UiBgButtonTrainAI.SetActive(false);
					UiBgButtonDriveAI.SetActive(false);
					UiBgButtonOptions.SetActive(false);
					UiBgButtonQuit.SetActive(true);
					break;
			}
			
			// check for inputs
			if (menuSelection < 4 && Input.GetKeyDown(KeyCode.DownArrow)) { menuSelection += 1; }
			if (menuSelection > 0 && Input.GetKeyDown(KeyCode.UpArrow)) { menuSelection -= 1; }

			if (Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.Return))
			{
				UiBgButtonDrive.SetActive(false);
				UiBgButtonTrainAI.SetActive(false);
				UiBgButtonDriveAI.SetActive(false);
				UiBgButtonOptions.SetActive(false);
				UiBgButtonQuit.SetActive(false);				

				// switch game mode
				if (menuSelection == 0) { Game.SwitchMode("driving"); }
				if (menuSelection == 1) { Game.SwitchMode("train_AI"); }
				if (menuSelection == 2) { Game.SwitchMode("drive_AI"); }
				//if (menuSelection == 3) { Game.SwitchMode("options"); }
				if (menuSelection == 4) { Application.Quit(); }
			}
		}
	}



}
