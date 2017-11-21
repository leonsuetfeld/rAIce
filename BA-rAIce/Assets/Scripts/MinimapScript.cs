using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text;
using System.Linq;
using UnityEngine.UI;


//http://answers.unity3d.com/questions/27968/getpixels-of-rendertexture.html
//https://stackoverflow.com/questions/32996534/rendering-complete-camera-view169-onto-a-texture-in-unity3d

public class MinimapScript : MonoBehaviour {

//	public Image pixel;
//	public Image pixel_clone;
//	public GameObject pixelParent;

	public GameScript Game;
	Rect showRect; Rect sendRect; Rect readRect;
	RenderTexture myRT;
	Texture2D myImg;
	Camera owncam;

	public float showRectXCoordinate;
	public int xPixels;
	public int yPixels;


	void Start() {
		owncam = gameObject.GetComponent<Camera> ();
		showRect = new Rect(showRectXCoordinate, 0.63f, 0.1f, 0.25f);
		owncam.rect = showRect;
	}


	public void PrepareVision(int xLen, int yLen) {
		xPixels = xLen;
		yPixels = yLen;
		owncam = gameObject.GetComponent<Camera> ();
		owncam.aspect = (xPixels + 0.0f) / yPixels; //0.5f;
		if (Game.AiInt.AIMode || Game.Rec.SV_SaveMode) {
			sendRect = new Rect (0, 0, 1, 1);
			readRect = new Rect (0, 0, xPixels, yPixels);
			myRT = new RenderTexture (xPixels, yPixels, 24);
			myRT.Create ();
			myImg = new Texture2D (xPixels, yPixels, TextureFormat.RGB24, false); //false = no mipmaps
		}
	}


	public string GetVisionDisplay() {
		if ((!Game.AiInt.AIMode && !Game.Rec.SV_SaveMode) || !Consts.usecameras) {
			return "";
		}

		Camera cam = gameObject.GetComponent<Camera> ();

		cam.rect = sendRect;
		cam.targetTexture = myRT;
		RenderTexture.active = myRT;
		try {
			cam.Render (); 
			RenderTexture.active = myRT;
			myImg.ReadPixels (readRect, 0, 0); //"the center section"
			myImg.Apply (false);

			//debug
			//		byte[] bytes;
			//		bytes = myImg.EncodeToPNG();
			//		System.IO.File.WriteAllBytes("./picpicpic.png", bytes );

			cam.targetTexture = null;
			RenderTexture.active = null;
			cam.rect = showRect;

		// return imgToArray(myImg); dann müsste man danach noch TwoDImageToStr aufrufen, aber das ist sinnlos
		return imgToStr(myImg, xPixels, yPixels);

		} catch (Exception e) {
			UnityEngine.Debug.Log ("Flare renderer to update not found - in UnityEngine.Camera:Render()");
			return "";
		}
	}



	public static string imgToStr(Texture2D myImg, int xLen, int yLen) {

		StringBuilder alltext = new StringBuilder((xLen*yLen)+xLen); //letztes für die kommas
		StringBuilder currline = new StringBuilder(yLen);
		try {
			if (Consts.SeeCurbAsOff) {
				
				for (int i = 0; i < myImg.width; i++) {
					currline = new StringBuilder(yLen);
					for (int j = 0; j < myImg.height; j++) {
						if ((float)myImg.GetPixel (i, j).grayscale > 0.8) //street
							currline.Append("1");
						else  											  //curb & off
							currline.Append("0");
					}
					//clinenr = ParseIntBase3 (currline);
					//alltext = alltext + clinenr.ToString("X") + ",";
					alltext.Append(currline+",");
				}

			} else {
				
				for (int i = 0; i < myImg.width; i++) {
					currline = new StringBuilder(yLen);
					for (int j = 0; j < myImg.height; j++) {
						if ((float)myImg.GetPixel (i, j).grayscale > 0.8) //street
							currline.Append("2");
						else if ((float)myImg.GetPixel (i, j).grayscale > 0.4) //curb
							currline.Append("1");
						else 											  //off
							currline.Append("0");
					}
					//clinenr = ParseIntBase3 (currline);
					//alltext = alltext + clinenr.ToString("X") + ",";
					alltext.Append(currline+",");
				}
			}
		} catch (Exception e) {
			UnityEngine.Debug.Log ("Converting the Camera-img to string went wrong :o");
		}
		return alltext.ToString();
	}




//	public static float[,] imgToArray(Texture2D myImg) {
//		float[,] visiondisplay = new float[myImg.width, myImg.height];
//
//		for (int i = 0; i < myImg.width; i++) {
//			for (int j = 0; j < myImg.height; j++) {
//				if (Consts.SeeCurbAsOff) {
//					if ((float)myImg.GetPixel (i, j).grayscale > 0.8) //street
//						visiondisplay [i, j] = 1;
//					else  											  //curb & off
//						visiondisplay [i, j] = 0;
//				} else {
//					if ((float)myImg.GetPixel (i, j).grayscale > 0.8) //street
//						visiondisplay [i, j] = 2;
//					else if ((float)myImg.GetPixel (i, j).grayscale > 0.4) //curb
//						visiondisplay [i, j] = 1;
//					else 											  //off
//						visiondisplay [i, j] = 0;
//				}
//			}
//		}
//		return visiondisplay;
//	}







//  Old way of prining the visiondisplay
//	public void CreatePixelImage(Image pixel, GameObject pixelParent, int xlen, int ylen)
//	{
//		for (int i = 0; i<xlen*ylen; i++)
//		{
//			int x = i%xlen;
//			int y = i/xlen;
//			pixel_clone = (Image)Instantiate(pixel, new Vector3(0,0,0), Quaternion.identity, pixelParent.transform);
//			pixel_clone.rectTransform.localScale = new Vector3(1f,1f,1f);
//			pixel_clone.rectTransform.localPosition = new Vector3(-28.5f+x*2.0f,-40.5f+y*2.0f,0);
//			pixel_clone.name = "visPixel_"+x.ToString()+"_"+y.ToString();
//		}
//	}
//
//	public void ShowVisionDisplay(float[,] a)
//	{
//		for (int x = 0; x<a.GetLength(0); x++)
//		{
//			for (int y = 0; y<a.GetLength(1); y++)
//			{
//				float c = a[x,y];
//				Image currentPixel = GameObject.Find("visPixel_"+x.ToString()+"_"+y.ToString()).GetComponent<Image>();
//				currentPixel.color = new Color (c,c,c);
//			}
//		}
//	}


}

