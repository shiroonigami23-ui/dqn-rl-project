using UnityEngine;

public class CinematicFollowCamera : MonoBehaviour
{
    public Transform target;
    public Vector3 offset = new Vector3(0f, 2.6f, -8f);
    public float smooth = 3.5f;
    public bool lookAtTarget = true;

    void LateUpdate()
    {
        if (target == null) return;

        Vector3 desired = target.position + offset;
        transform.position = Vector3.Lerp(transform.position, desired, Time.deltaTime * smooth);

        if (lookAtTarget)
        {
            Vector3 lookPoint = target.position + new Vector3(0f, 0.5f, 0f);
            Quaternion rot = Quaternion.LookRotation(lookPoint - transform.position);
            transform.rotation = Quaternion.Slerp(transform.rotation, rot, Time.deltaTime * smooth);
        }
    }
}
