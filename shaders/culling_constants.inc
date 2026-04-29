/*

常量定义文件，包含 culling 相关的所有常量

*/

// 一个极小值，用于消除除以 0 的错误（例如当 w 分量接近 0 时）
const float c_epsilon    = 1.2e-07f;
// 深度微调量。在深度比较时给一个点深度值，防止由于浮点精度问题导致物体自己遮挡自己（Z-Fighting）
// 这里取的是 24 位精度缓冲区的两个量化步长
const float c_depthNudge = 2.0/float(1<<24);
