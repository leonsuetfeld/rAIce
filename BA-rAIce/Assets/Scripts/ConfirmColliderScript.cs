using UnityEngine;
using System.Collections;

public class ConfirmColliderScript : MonoBehaviour {

	public TimingScript Timing;

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
	
	}

	void OnTriggerExit (Collider other)
	{
		Timing.FlipCcPassed();
		Timing.justResettet = false;
	}
}
