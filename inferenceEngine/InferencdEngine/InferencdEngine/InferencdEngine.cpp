#include "InferencedEngine.h"
#include "cpp/ie_plugin_cpp.hpp"
#include "details/ie_so_pointer.hpp"
#include "ie_plugin.hpp"
#include "ie_ihetero_plugin.hpp"
#include "ie_extension.h"
#include <ie_plugin_dispatcher.hpp>
#include <string>
#include <vector>
#include <map>
#include <ie_api.h>
#include <ie_common.h>
#include "cpp/ie_cnn_net_reader.h"
#include "ie_plugin_config.hpp"
#include "inference_engine.hpp"
using namespace std;
using namespace InferenceEngine;

#define DEBUG_OPENVINO_ECG_PRINT_ERR(format,...)  printf(format,##__VA_ARGS__)
#define DEBUG_OPENVINO_ECG_PRINT(format,...)  printf(format,##__VA_ARGS__)
class Ecg_openvino {
public:
	ExecutableNetwork executable_network;
	InferRequest infer_request;
	CNNNetwork network;
	int batchSize;
};
static map<string, Ecg_openvino> pbname2sessionptr;



int loadGraphWithLicenceAndPbnameBatchsize(char *model_xml_path, char *model_bin_path,float per_process_gpu_memory_fraction, int batchSizeParam)
{
	if (model_xml_path ==NULL) {
		DEBUG_OPENVINO_ECG_PRINT_ERR("xml_path is NULL");
		return -1;
	}
	if (model_bin_path == NULL) {
		DEBUG_OPENVINO_ECG_PRINT_ERR("bin_path is NULL");
		return -1;
	}
	try {
		//1 load a plugin
		vector<file_name_t> pluginDirs;
		string s("/home/ubuntu/create_so/inference_engine/lib/intel64/");
		file_name_t ws;
		ws.assign(s.begin(), s.end());
		pluginDirs.push_back(ws);
		InferenceEnginePluginPtr engine_ptr = PluginDispatcher(pluginDirs).getSuitablePlugin(TargetDevice::eGPU);
		InferencePlugin plugin(engine_ptr);


		//2 read a model IR 
		CNNNetReader network_reader;
		network_reader.ReadNetwork(model_xml_path);
		network_reader.ReadWeights(model_bin_path);
		auto network = network_reader.getNetwork();


		//3 configure input and output format
		InputsDataMap input_info = network.getInputsInfo();
		if (signed(input_info.size()) != 1) throw std::logic_error("no inputs info is provided");
		if (input_info.empty()) throw std::logic_error("supports topologies only with 1 input");
		auto inputInfoItem = *input_info.begin();
		inputInfoItem.second->setPrecision(Precision::FP32);
		inputInfoItem.second->setLayout(Layout::NC);

		//prepare output blobs
		OutputsDataMap output_info(network.getOutputsInfo());
		string firstOutputName;
		for (auto &item : output_info) {
			if (firstOutputName.empty()) {
				firstOutputName = item.first;
			}
			DataPtr output_data = item.second;
			if (!output_data)
			{
				throw std::logic_error("output data pointer is not valid");
			}
			output_data->setPrecision(Precision::FP32);
		}
		const SizeVector outputDims = output_info.begin()->second->getDims();
		if (batchSizeParam > 0)
			network.setBatchSize(batchSizeParam);
		size_t batchSize = network.getBatchSize();


		//4 load the model to the plugin
		Ecg_openvino ecg_network;
		std::map<std::string, std::string> config = { { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } };
		auto executable_network = plugin.LoadNetwork(network, config);
		inputInfoItem.second = {};
		output_info = {};
		network = {};
		network_reader = {};
		//5 create an infer request
		auto infer_request = executable_network.CreateInferRequest();
		ecg_network.executable_network = executable_network;
		ecg_network.infer_request = infer_request;
		ecg_network.batchSize = int(batchSize);
		string model_xml_path_str = string(model_xml_path);
		string pbname = model_xml_path_str.substr(0, model_xml_path_str.find("."));
		pbname2sessionptr.insert(pair<string, Ecg_openvino>(pbname, ecg_network));
		return 1;
	}
	catch(exception e){
		DEBUG_OPENVINO_ECG_PRINT_ERR("load model is error");
		return -1;
	}
}

static void do_classify_with_one_batch_size_pbname(char* pbname_c, float *data, int width, int batchsize, int *out_label, float *out_score, int data_offset, int label_offset, int num_class, char* input_layer_c, char*output_layer_c)
{
	//first we load and initialize the model
	string pbname_st(pbname_c);
	string pbname = pbname_st.substr(0, pbname_st.find("."));
	string input_layer(input_layer_c);
	string output_layer(output_layer_c);

	//executeable network executeable_network = pbname2sessionptr[name]
	Ecg_openvino ecg_network = pbname2sessionptr[pbname];
	int real_batchsize = ecg_network.batchSize;

	//prepare our input
	ConstInputsDataMap inputInfo = ecg_network.executable_network.GetInputsInfo();
	for (const auto &item : inputInfo)
	{
		Blob::Ptr input = ecg_network.infer_request.GetBlob(item.first);
		//fill input tensor with images,first b channel,then g and r channels
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];
		auto tempdata = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
		memset(tempdata,0,real_batchsize*image_size*sizeof(float));
		memcpy(tempdata,data,batchsize*image_size*sizeof(float));
	}
	inputInfo = {};
	
	//do inference
	ecg_network.infer_request.Infer();

	//process output
	ConstOutputsDataMap outputInfo = ecg_network.executable_network.GetOutputsInfo();
	string firstoutputName;
	for (auto &item : outputInfo) {
		firstoutputName = item.first;
		break;
	}
	const Blob::Ptr output_blob = ecg_network.infer_request.GetBlob(firstoutputName);
	auto output_data = output_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

	//this vector stores id's of top N results
	vector<unsigned> results;
	int FLAGS_nt = 1;//num_class
	//validating -nt value
	const int resultsCnt = int(output_blob->size() / real_batchsize);
	if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
		DEBUG_OPENVINO_ECG_PRINT_ERR("FLAGS_nt is not available for this network (-nt should be less than resultsCnt + 1 and more than 0) FLAGS_nt will be used maximal value : ");
		FLAGS_nt = resultsCnt;
	}
	
	TopResults(FLAGS_nt, *output_blob, results);
	DEBUG_OPENVINO_ECG_PRINT("##FLAGS_nt = %d,output_blob->size()=%lu,resultsCnt=%d,real_batchsize=%d\n", FLAGS_nt, output_blob->size(), resultsCnt, real_batchsize);

	//print the result iterating over each batch
	for (int image_id = 0; image_id < batchsize; image_id++) {
		const auto resulttemp = output_data[results[image_id] + image_id*(output_blob->size() / real_batchsize)];
		out_label[image_id] = results[image_id];
		out_score[image_id] = resulttemp;
		cout.precision(7);
		cout << left << fixed << results[image_id] << " " << out_score[image_id] << endl;
	}
}


//在使用模型之前，先将模型进行加载
#ifdef _WIN32
int _stdcall loadGraphWithLicenceAndPbname(char *xml_path, char *bin_path, float per_process_gpu_memory_fraction)
#else
int loadGraphWithLicenceAndPbname(char *xml_path, char *bin_path, float per_process_gpu_memory_fraction)
#endif
{
	//将licence去掉，使用原始的加载方式。
	return loadGraphWithLicenceAndPbnameBatchsize(xml_path, bin_path, per_process_gpu_memory_fraction, -1);
}


//在使用的过程中，使用xml的名字，不包括后缀
#ifdef _WIN32
void _stdcall do_classify_with_pbname(char* pbname_c, float *data, int sample_num, int width, int batch_size, int * out_label, float * out_score, int num_class, char* input_layer_c, char* output_layer_c)
#else
void do_classify_with_pbname(char* pbname_c, float *data, int sample_num, int width, int batch_size, int * out_label, float * out_score, int num_class, char* input_layer_c, char* output_layer_c)
#endif
{
	Ecg_openvino ecg_network;
	if ((pbname_c == NULL) || (data == NULL) || (out_label == NULL) || (out_score == NULL) || (input_layer_c == NULL) || (output_layer_c == NULL))
	{
		DEBUG_OPENVINO_ECG_PRINT_ERR("pbname_c is %p data is %p out_label is %p out_score is %p input_layer_c is %p output_layer_c is %p\n", pbname_c, data, out_label, out_score, input_layer_c, output_layer_c);
		return;
	}

	string pbname(pbname_c);
	ecg_network = pbname2sessionptr[pbname];
	int real_batch_size = ecg_network.batchSize;
	
	//计算要进行多少次循环
	int count = sample_num / real_batch_size;
	int data_offset = 0;
	int label_offset = 0;
	int out_score_offset = 0;
	int out_layer_c_offset = 0;
	for (int i = 0; i < count; i++) {
		data_offset = i*real_batch_size*width;
		label_offset = i*real_batch_size*width;
		out_score_offset = i*real_batch_size;
		out_layer_c_offset = i*real_batch_size;
		do_classify_with_one_batch_size_pbname(pbname_c,data+data_offset,width,real_batch_size,out_label+label_offset,out_score+out_score_offset,data_offset,label_offset,num_class,input_layer_c,output_layer_c);
	}

	int leftsample = sample_num%batch_size;
	if (leftsample > 0) {
		data_offset = count*real_batch_size*width;
		label_offset = count*real_batch_size*width;
		out_score_offset = count*real_batch_size;
		out_layer_c_offset = count*real_batch_size;
		do_classify_with_one_batch_size_pbname(pbname_c, data + data_offset, width, real_batch_size, out_label + label_offset, out_score + out_score_offset, data_offset, label_offset, num_class, input_layer_c, output_layer_c);
	}

}

//将pbname进行清空
#ifdef _WIN32
void _stdcall unloadGraph_with_pbnanme(char *pbname_c)
#else
void  unloadGraph_with_pbnanme(char *pbname_c)
#endif
{
	string pbname_str(pbname_c);
	string pbname = pbname_str.substr(0, pbname_str.find("."));
	if (pbname2sessionptr.count(pbname) > 0) {
		pbname2sessionptr.erase(pbname);
	}
	return;
}


