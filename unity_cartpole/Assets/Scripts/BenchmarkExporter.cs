using System.IO;
using UnityEngine;
using System.Diagnostics;

public class BenchmarkExporter : MonoBehaviour
{
    public CartPoleArenaManager manager;
    public ReplayCapture capture;
    public int autoExportEveryNEpisodes = 5;

    private string csvPath;

    void Start()
    {
        csvPath = Path.Combine(Application.dataPath, "..", "benchmark_log.csv");
        if (!File.Exists(csvPath))
            File.WriteAllText(csvPath, "episode,dqn_reward,random_reward,dqn_steps,random_steps\n");

        if (manager != null)
            manager.OnEpisodeFinished += OnEpisodeFinished;
    }

    void OnDestroy()
    {
        if (manager != null)
            manager.OnEpisodeFinished -= OnEpisodeFinished;
    }

    void OnEpisodeFinished(int ep, float dqnReward, float rndReward, int dqnSteps, int rndSteps)
    {
        File.AppendAllText(csvPath, $"{ep},{dqnReward:F2},{rndReward:F2},{dqnSteps},{rndSteps}\n");

        if (capture != null)
        {
            if ((ep + 1) % autoExportEveryNEpisodes == 0)
            {
                capture.captureEnabled = false;
                TryExportVideo();
                capture.captureEnabled = true;
            }
        }
    }

    void TryExportVideo()
    {
        string root = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        string script = Path.Combine(root, "convert_frames_to_video.py");
        if (!File.Exists(script)) return;

        try
        {
            var p = new Process();
            p.StartInfo.FileName = "python";
            p.StartInfo.Arguments = $"\"{script}\" --frames_dir ReplayFrames --out_mp4 replay_auto.mp4 --out_gif replay_auto.gif --fps 30";
            p.StartInfo.WorkingDirectory = root;
            p.StartInfo.CreateNoWindow = true;
            p.StartInfo.UseShellExecute = false;
            p.Start();
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogWarning($"Auto export failed: {e.Message}");
        }
    }
}
