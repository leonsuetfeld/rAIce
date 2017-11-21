using UnityEngine;
using System.Collections;

public class TimingScript : MonoBehaviour {

	// variables for lap time measurement
	public float currentLapTime;
	public float lastLapTime;
	public float fastestLapTime;
	public int fastestLapCount;
	float currentLapStart;
	public int lapCount = 0;
	public bool timeSet = false;
	public bool fastLapSet = false;
	public bool activeLap;
	public bool ccPassed = false;
	public UIScript UserInterface;
	public CarController Car;
	public bool justResettet = false;

	// variables for lapSaving
	public Recorder Rec;

	private float time_punishs;

	// Use this for initialization
	void Start () 
	{
		time_punishs = 0;	
	}
	
	// not sure if this is supposed to be in update or fixedupdate.
	void FixedUpdate () 
	{
		if (activeLap)
		{
			currentLapTime = (Time.time - currentLapStart) + time_punishs;
		}
	}

	// Start/ finish collider updating the laptimes
	void OnTriggerExit (Collider other)            //"this" here is the timingsystem, a collider in root, and the only "other" there is that can move is only the car
	{
		if (!justResettet) { //wenn das hier nicht ist kann der sich nach wallhit-reset direkt hiervor projizieren und wäre done, which is wrong
			justResettet = false;
			// last lap time & fastest lap time update 
			if (activeLap && ccPassed) {  //ccpassed heißt dass er schon durch den zweiten collider ist, lapclean heißt 1 reifen auf straße... activelap ist true sobald man den trigger entered (was dank CarControllerSkript nur im game-modus geht)//also, im grunde kommt man hier schon rein wenn man eine valide, ungecheatete, komplette runde gefahren ist.
				if (Car.lapClean) {
					lastLapTime = Time.time - currentLapStart + time_punishs;
					timeSet = true;
					if (!fastLapSet || (timeSet && lastLapTime < fastestLapTime)) { //wenn diese die erste oder letzte runde ist
						fastestLapTime = lastLapTime;
						fastestLapCount = lapCount;
						fastLapSet = true;
					}
					Rec.FinishList ();
				}
				if (Car.AiInt.AIMode) {
					Car.AiInt.EndRound (Car.lapClean);
				}
			}
			Start_Round();
		}
	}

	public void Start_Round()
	{
		activeLap = true; //activelap ists erst nach dem zweiten validen ungecheateten colliderdurchlauf
		Rec.StartList ();
		lapCount += 1;
		currentLapStart = Time.time;
		Car.LapCleanTrue ();
		ccPassed = false; //wird erst wieder true wenn man durch den zweiten collider ist, und wieder falls sobald man cheatenderweise nochmal anschließend zurückfährt (deswegen da ontriggerexit!)
		time_punishs = 0;
	}

	public void Stop_Round() 
	{
		justResettet = false;
		Car.LapCleanTrue ();
		ccPassed = false;
		currentLapStart = Time.time;
		ResetTiming ();
		Rec.Car.Game.UserInterface.DrivingOverlayHandling ();
	}


	public void PunishTime(float howmuch) 
	{
		if (activeLap)
			time_punishs += howmuch;
	}


	// Reset Timing Script
	public void ResetTiming()
	{
		currentLapTime = 0.0f;
		activeLap = false;
		time_punishs = 0;
	}

	// Reset Session Script
	public void ResetSessionTiming()
	{
		lapCount = 0;
		currentLapTime = 0.0f;
		timeSet = false;
		activeLap = false;
		time_punishs = 0;
	}

	public void FlipCcPassed()
	{
		if (ccPassed) { ccPassed = false; }
		else if (!ccPassed) { ccPassed = true; }
	}

}