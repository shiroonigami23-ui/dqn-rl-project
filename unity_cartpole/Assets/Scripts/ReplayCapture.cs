using System.IO;
using UnityEngine;

public class ReplayCapture : MonoBehaviour
{
    public bool captureEnabled = false;
    public int captureEveryNFrames = 2;
    public int superSize = 1;
    public string folderName = "ReplayFrames";

    private int shotId = 0;
    private string outDir;

    void Start()
    {
        outDir = Path.Combine(Application.dataPath, "..", folderName);
        if (!Directory.Exists(outDir)) Directory.CreateDirectory(outDir);
        Debug.Log($"Replay output dir: {outDir}");
    }

    public int FrameCount => shotId;

    public void ResetCounter()
    {
        shotId = 0;
    }

    void LateUpdate()
    {
        if (!captureEnabled) return;
        if (Time.frameCount % captureEveryNFrames != 0) return;

        string file = Path.Combine(outDir, $"frame_{shotId:000000}.png");
        ScreenCapture.CaptureScreenshot(file, superSize);
        shotId++;
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.box);
        style.fontSize = 14;
        GUI.Box(new Rect(Screen.width - 300, 10, 290, 84),
            $"Capture: {(captureEnabled ? "ON" : "OFF")}\nFrames: {shotId}\nKeys: C toggle", style);

        if (Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.C)
        {
            captureEnabled = !captureEnabled;
        }
    }
}
