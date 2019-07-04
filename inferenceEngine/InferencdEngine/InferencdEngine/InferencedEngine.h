#pragma once

#ifdef _WIN32
extern "C"
{
	__declspec(dllexport) int _stdcall loadGraphWithLicenceAndPbname(char *xml_path, char *bin_path, float per_process_gpu_memory_fraction);
	__declspec(dllexport) void _stdcall do_classify_with_pbname(char* pbname_c, float *data, int sample_num, int width, int batch_size, int * out_label, float * out_score, int num_class, char* input_layer_c, char* output_layer_c);
	__declspec(dllexport) void _stdcall unloadGraph_with_pbnanme(char *pbname_c);
}
#else
#ifdef __cplusplus
extern "C"
{
#endif
	int  loadGraphWithLicenceAndPbname(char *xml_path, char *bin_path, float per_process_gpu_memory_fraction);
	void do_classify_with_pbname(char* pbname_c, float *data, int sample_num, int width, int batch_size, int * out_label, float * out_score, int num_class, char* input_layer_c, char* output_layer_c);
	void unloadGraph_with_pbnanme(char *pbname_c);
#ifdef  __cplusplus
}
#endif 
#endif // _WIN32

