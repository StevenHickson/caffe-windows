#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"
#include "caffe/util/io.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#endif

//void segmentation(cv::Mat &in, cv::Mat &out) {
//	cv::Mat imgGray;
//	cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
//	std::vector<std::vector<cv::Point> > contours;
//	std::vector<cv::Vec4i> hierarchy;
//	cv::threshold(imgGray, imgGray, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
//	cv::findContours(imgGray, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
//	cv::Mat markers(img.size(), CV_32S);
//	markers = cv::Scalar::all(0);
//	int compCount = 0;
//	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], compCount++)
//		cv::drawContours(markers, contours, idx, cv::Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
//	cv::watershed(img, markers);
//}


/* BEGIN SLIC */
/* -----------------------------------------------------------------------------------------*/
struct SLIC_Center{
	float x, y, L, A, B;
	bool alive;
};

class SLIC{
public:
	SLIC(int width, int height);
	void ComputeSuperPixels(const cv::Mat &image, cv::Point &topLeft, int S, int M, int iter);
	void DrawResults(cv::Mat &display);
	void ApplyMask(const cv::Mat &mask, cv::Mat &display);
	int Width(){ return m_width; }
	int Height(){ return m_height; }

	float m_spThresh;

private:
	void ConvertToLAB();
	void Initialize();
	void AssignPixels();
	void UpdateCenters();

	std::vector<SLIC_Center>	m_centers;
	cv::Mat						m_lab;  //lab
	cv::Mat						m_results;  //int
	cv::Mat						m_dist;  //float
	cv::Mat						m_input;  //rgb
	cv::Point					m_topLeft;

	int							m_width;
	int							m_height;
	int							m_S;
	int							m_M;
	float						m_scale;
	float						m_scale2;
	int							m_iter;
};

SLIC::SLIC(int width, int height){
	m_lab = cv::Mat(height, width, CV_8UC3);
	m_results = cv::Mat::zeros(height, width, CV_32SC1);
	m_dist = cv::Mat(height, width, CV_32FC1);

	m_spThresh = 0.5f; //default

	m_width = width;
	m_height = height;
}

void SLIC::ConvertToLAB(){
	cv::cvtColor(m_input, m_lab, CV_BGR2Lab);
}

void SLIC::Initialize(){
	//memset(m_results.data, 0xFF, 4 * m_results.rows * m_results.cols);


	// create regularized grid
	for (int i = m_S / 2; i < m_height; i += m_S){
		for (int j = m_S / 2; j < m_width; j += m_S){
			cv::Vec3b lab = m_lab.at<cv::Vec3b>(i, j);
			m_centers.push_back(SLIC_Center{ j, i, lab[0], lab[1], lab[2], true });

			for (int k = i - m_S / 2; k < i + m_S / 2 && k < m_height; k++){
				if (k < 0){
					continue;
				}
				for (int l = j - m_S / 2; l < j + m_S / 2 && l < m_width; l++){
					if (l < 0){
						continue;
					}

					m_results.at<int>(k, l) = m_centers.size() - 1;
				}
			}
		}
	}
}

void SLIC::AssignPixels(){
	// reset distances
	//memset(m_dist.data, std::numeric_limits<float>::max(), m_dist.rows * m_dist.cols);
	cv::Mat_<float>::iterator pF = m_dist.begin<float>();
	while (pF != m_dist.end<float>()) {
		*pF++ = std::numeric_limits<float>::max();
	}

	// loop over region of each center
	std::vector<SLIC_Center>::iterator pC = m_centers.begin();
	for (int i = 0; i < m_centers.size(); ++i, ++pC) {
		if (pC->alive == false){
			continue;
		}

		float x = pC->x;
		float y = pC->y;

		cv::Vec3b center;
		center[0] = pC->L;
		center[1] = pC->A;
		center[2] = pC->B;

		for (int row = y - m_S; row < y + m_S && row < m_height; row++){
			if (row < 0){
				continue;
			}

			for (int col = x - m_S; col < x + m_S && col < m_width; col++){
				if (col < 0){
					continue;
				}

				// compute the color distance
				cv::Vec3b pix = m_lab.at<cv::Vec3b>(row, col);
				float colorDist = std::sqrtf(
					(int(pix[0]) - int(center[0])) * (int(pix[0]) - int(center[0])) +
					(int(pix[1]) - int(center[1])) * (int(pix[1]) - int(center[1])) +
					(int(pix[2]) - int(center[2])) * (int(pix[2]) - int(center[2]))
					);

				//diff between x,y and col,row
				if (colorDist < m_dist.at<float>(row, col)){
					m_dist.at<float>(row, col) = colorDist;
					m_results.at<int>(row, col) = i;
				}
			}
		}
	}
}

void SLIC::UpdateCenters(){
	int *newX = (int *)malloc(m_centers.size() * sizeof(int));
	int *newY = (int *)malloc(m_centers.size() * sizeof(int));
	int *L = (int *)malloc(m_centers.size() * sizeof(int));
	int *A = (int *)malloc(m_centers.size() * sizeof(int));
	int *B = (int *)malloc(m_centers.size() * sizeof(int));
	int *counter = (int *)malloc(m_centers.size() * sizeof(int));

	memset(newX, 0, m_centers.size() * sizeof(int));
	memset(newY, 0, m_centers.size() * sizeof(int));
	memset(L, 0, m_centers.size() * sizeof(int));
	memset(A, 0, m_centers.size() * sizeof(int));
	memset(B, 0, m_centers.size() * sizeof(int));
	memset(counter, 0, m_centers.size() * sizeof(int));


	cv::Mat_<cv::Vec3b>::const_iterator pLAB = m_lab.begin<cv::Vec3b>();
	cv::Mat_<int>::const_iterator pR = m_results.begin<int>();
	for (int i = 0; i < m_height; i++){
		for (int j = 0; j < m_width; j++, pR++, pLAB++){

			if (*pR < m_centers.size()){
				newX[*pR] += j;
				newY[*pR] += i;

				cv::Vec3b lab = *pLAB;// RGB2LAB(image[index].red, image[index].green, image[index].blue);

				L[*pR] += lab[0];
				A[*pR] += lab[1];
				B[*pR] += lab[2];
				counter[*pR]++;
			}
		}
	}

	int *pI = counter, *pX = newX, *pY = newY, *pL = L, *pA = A, *pB = B;
	std::vector<SLIC_Center>::iterator pC = m_centers.begin();
	while (pC != m_centers.end()) {

		if (*pI == 0){
			pC->alive = false;
			pC->x = -99999.0f;
			pC->y = -99999.0f;
		}
		else{
			pC->x = (float)*pX / (float)*pI;
			pC->y = (float)*pY / (float)*pI;
			pC->L = *pL / (float)*pI;
			pC->A = *pA / (float)*pI;
			pC->B = *pB / (float)*pI;
		}
		++pI; ++pC; ++pX; ++pY; ++pL; ++pA; ++pB;
	}
}

void SLIC::DrawResults(cv::Mat &display){
	//set display = m_input
	display = m_input.clone();

	int safeHeight = m_height - 1, safeWidth = m_width - 1;
	cv::Mat_<cv::Vec3b>::iterator pD = display.begin<cv::Vec3b>();
	cv::Mat_<int>::const_iterator pR = m_results.begin<int>();
	for (int i = 0; i < m_height; i++) {
		for (int j = 0; j < m_width; j++, pD++, pR++) {

			if (i < safeHeight && j < safeWidth && (*pR != *(pR + 1) || *pR != *(pR + m_width))) {
				(*pD)[0] = 0;
				(*pD)[1] = 0;
				(*pD)[2] = 255;
			}
		}
	}

	cv::Vec3b red = cv::Vec3b(0, 0, 255);
	std::vector<SLIC_Center>::iterator pC = m_centers.begin();
	for (int i = 0; i < m_centers.size(); i++, pC++) {
		if (pC->alive == false) {
			continue;
		}

		//int index = (int)m_centers[i].y * m_width + (int)m_centers[i].x;

		int y = pC->y, x = pC->x, safeHeight = m_height - 1;
		if (x > 1 && y > 1 && x <  safeWidth && y < safeHeight){
			display.at<cv::Vec3b>(y, x) = red;

			display.at<cv::Vec3b>(y, x + 1) = red;

			display.at<cv::Vec3b>(y, x - 1) = red;

			display.at<cv::Vec3b>(y + 1, x) = red;

			display.at<cv::Vec3b>(y - 1, x) = red;
		}
	}
}

void SLIC::ApplyMask(const cv::Mat &mask, cv::Mat &display) {
	//I need to go through the results image and figure out which clusters are fully/mostly contained by the mask and then output the display

	int *good = (int *)calloc(m_centers.size(), sizeof(int));
	int *bad = (int *)calloc(m_centers.size(), sizeof(int));
	bool *accepted = (bool *)calloc(m_centers.size(), sizeof(bool));
	//iterate through the image and grab the good and bad values
	cv::Mat_<unsigned char>::const_iterator pM = mask.begin<unsigned char>();
	cv::Mat_<int>::const_iterator pR = m_results.begin<int>();
	while (pM != mask.end<unsigned char>()) {
		if (*pM != 255)
			good[*pR]++;
		else
			bad[*pR]++;
		++pR; ++pM;
	}

	float minusThresh = 1.0f - m_spThresh;
	//lets see what variables we want
	int *pG = good, *pB = bad;
	bool *pA = accepted;
	for (int i = 0; i < m_centers.size(); i++) {
		if (*pG != 0 && minusThresh * float(*pG) >= m_spThresh * float(*pB))
			*pA = true;
		else
			*pA = false;
		++pG; ++pA; ++pB;
	}

	//now lets display only those superpixels
	//method displaying superpixels using mask
	/*display = cv::Mat::zeros(m_input.rows, m_input.cols, CV_8UC3);
	cv::Mat_<cv::Vec3b>::iterator pD = display.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator pI = m_input.begin<cv::Vec3b>();
	pR = m_results.begin<int>();
	while (pD != display.end<cv::Vec3b>()) {
	if (accepted[*pR])
	*pD = *pI;
	++pR; ++pD; ++pI;
	}*/

	//method displaying mask using superpixels
	display = cv::Mat::zeros(m_input.rows, m_input.cols, CV_8UC3);
	cv::Mat_<cv::Vec3b>::iterator pD = display.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator pI = m_input.begin<cv::Vec3b>();
	pR = m_results.begin<int>();
	while (pD != display.end<cv::Vec3b>()) {
		if (*pR != 255 && accepted[*pR])
			*pD = *pI;
		++pR; ++pD; ++pI;
	}
}

void SLIC::ComputeSuperPixels(const cv::Mat &image, cv::Point &topLeft, int S, int M, int iter){
	m_topLeft = topLeft;
	//m_S = int(float(m_width)*float(S) * 0.01f);
	//m_M = int(float(m_width)*float(M) * 0.01f);
	m_S = S;
	m_M = M;
	m_scale = (float)M / (float)S;
	m_input = image;
	m_iter = iter;

	this->ConvertToLAB();
	this->Initialize();
	for (int i = 0; i < iter; i++){
		this->AssignPixels();
		this->UpdateCenters();
	}
}


/* END SLIC */
/* -----------------------------------------------------------------------------------------*/


/* BEGIN FH */
/* -----------------------------------------------------------------------------------------*/

#define WIDTH 4.0
#define THRESHOLD(size, c) (c/size)
template <class T>
inline T square(const T &x) { return x*x; };

/* make filters */
#define MAKE_FILTER(name, fun)                                \
	std::vector<float> make_ ## name(float sigma)       \
{                                                           \
	sigma = std::max(sigma, 0.01F);			                            \
	int len = (int)std::ceil(sigma * WIDTH) + 1;                     \
	std::vector<float> mask(len);                               \
for (int i = 0; i < len; i++)                               \
{                                                           \
	mask[i] = fun;                                              \
}                                                           \
	return mask;                                                \
}

MAKE_FILTER(fgauss, (float)expf(-0.5*square(i / sigma)));

class Edge {
public:
	float w;
	int a, b;
	bool valid;
	Edge() : w(0), a(0), b(0), valid(false) { };

	bool operator< (const Edge &other) const {
		return (w < other.w);
	}
};

typedef struct
{
	int rank;
	int p;
	int size;
} uni_elt;

class Universe
{
public:
	Universe() : num(0) { }
	Universe(int elements)
	{
		num = elements;
		elts.resize(num);
		std::vector<uni_elt>::iterator p = elts.begin();
		int i = 0;
		while (p != elts.end()) {
			p->rank = 0;
			p->size = 1;
			p->p = i;
			p++;
			i++;
		}
	}
	~Universe(){};
	int find(int x)
	{
		int y = x;
		while (y != elts[y].p)
			y = elts[y].p;
		elts[x].p = y;
		return y;
	};
	void join(int x, int y)
	{
		if (elts[x].rank > elts[y].rank)
		{
			elts[y].p = x;
			elts[x].size += elts[y].size;
		}
		else
		{
			elts[x].p = y;
			elts[y].size += elts[x].size;
			if (elts[x].rank == elts[y].rank)
				elts[y].rank++;
		}
		num--;
	}
	void release() {
		elts.clear();
	}
	int size(int x) const { return elts[x].size; }
	int num_sets() const { return num; }
	//should be private but I need to access some things
	std::vector<uni_elt>elts;
	int num;
};


void normalize(std::vector<float> &mask)
{
	int len = mask.size();
	float sum = 0;
	int i;
	for (i = 1; i < len; i++)
	{
		sum += std::fabsf(mask[i]);
	}
	sum = 2 * sum + std::fabsf(mask[0]);
	for (i = 0; i < len; i++)
	{
		mask[i] /= sum;
	}
}

/* convolve src with mask.  dst is flipped! */
void convolve_even(const cv::Mat& src, cv::Mat &dst, std::vector<float> &mask)
{
	int width = src.cols;
	int height = src.rows;
	int len = mask.size();
	dst = cv::Mat(src.rows, src.cols, src.type());
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			//cout << x << ", " << y << endl;
			float sum = mask[0] * src.at<float>(y, x);
			for (int i = 1; i < len; i++) {
				sum += mask[i] * (src.at<float>(y, std::max(x - i, 0)) + src.at<float>(y, std::min(x + i, width - 1)));
			}
			dst.at<float>(y, x) = sum;
		}
	}
}

void iExtractRGBColorSpace(const cv::Mat& in, cv::Mat &B, cv::Mat &G, cv::Mat &R) {
	B = cv::Mat(in.rows, in.cols, CV_32F);
	G = cv::Mat(in.rows, in.cols, CV_32F);
	R = cv::Mat(in.rows, in.cols, CV_32F);
	cv::Mat_<cv::Vec3b>::const_iterator pI = in.begin<cv::Vec3b>();
	cv::Mat_<float>::iterator pB = B.begin<float>(), pG = G.begin<float>(), pR = R.begin<float>();
	while (pI != in.end<cv::Vec3b>()) {
		*pB = (float)(*pI)[0];
		*pG = (float)(*pI)[1];
		*pR = (float)(*pI)[2];
		pI++; pB++; pG++; pR++;
	}
}

void iSmooth(const cv::Mat &src, float sigma, cv::Mat &out) {
	std::vector<float> mask = make_fgauss(sigma);
	normalize(mask);
	cv::Mat tmp(src.rows, src.cols, src.type());
	convolve_even(src, tmp, mask);
	convolve_even(tmp, out, mask);
}

void iBuildGraph(const std::vector<cv::Mat> &in,
	float sigma,
	Edge *&edges,
	int *num_edges)
{
	int width = in[0].cols;
	int height = in[0].rows;
	int num = 0;
	int x, y, xp, ym, yp;
	int safeWidth = width - 1, safeHeight = height - 1;
	int reserve_size = in[0].rows * in[0].cols * 8;
	//printf("Reserve size = %d\n",reserve_size);
	edges = (Edge*)malloc(reserve_size*sizeof(Edge));
	if (edges == NULL) {
		printf("Error, could not malloc\n");
		return;
	}
	cv::Mat smooth_r, smooth_g, smooth_b;
	//iExtractRGBColorSpace(in, B, G, R);
	iSmooth(in[0], sigma, smooth_b);
	iSmooth(in[1], sigma, smooth_g);
	iSmooth(in[2], sigma, smooth_r);

	//Normalize

	Edge *p = edges;
	cv::Mat_<float>::const_iterator pR = smooth_r.begin<float>(), pG = smooth_g.begin<float>(), pB = smooth_b.begin<float>();
	cv::Mat_<float>::const_iterator pRBegin = pR, pGBegin = pG, pBBegin = pB;
	for (y = 0, ym = -1, yp = 1; y < height; y++, ym++, yp++)
	{
		for (x = 0, xp = 1; x < width; x++, xp++)
		{
			//cout << x << ", " << y << endl;
			if (x < safeWidth)
			{
				Edge edge;
				edge.a = y * width + x;
				edge.b = y * width + xp;
				edge.w = sqrtf(square(*pR - *(pRBegin + edge.b)) + square(*pG - *(pGBegin + edge.b)) + square(*pB - *(pBBegin + edge.b)));
				//edge.valid = true;
				*p++ = edge;
				num++;
			}
			if (y < safeHeight)
			{
				Edge edge;
				edge.a = y * width + x;
				edge.b = yp * width + x;
				edge.w = sqrtf(square(*pR - *(pRBegin + edge.b)) + square(*pG - *(pGBegin + edge.b)) + square(*pB - *(pBBegin + edge.b)));
				//edge.valid = true;
				*p++ = edge;
				num++;
			}
			if ((x < safeWidth) && (y < safeHeight))
			{
				Edge edge;
				edge.a = y * width + x;
				edge.b = yp * width + xp;
				edge.w = sqrtf(square(*pR - *(pRBegin + edge.b)) + square(*pG - *(pGBegin + edge.b)) + square(*pB - *(pBBegin + edge.b)));
				//edge.valid = true;
				*p++ = edge;
				num++;
			}
			if ((x < safeWidth) && (y > 0))
			{
				Edge edge;
				edge.a = y * width + x;
				edge.b = ym * width + xp;
				edge.w = sqrtf(square(*pR - *(pRBegin + edge.b)) + square(*pG - *(pGBegin + edge.b)) + square(*pB - *(pBBegin + edge.b)));
				//edge.valid = true;
				*p++ = edge;
				num++;
			}
			pR++; pG++; pB++;
		}
	}
	smooth_b.release();
	smooth_g.release();
	smooth_r.release();
	*num_edges = num;
}

bool lessThan(const Edge& a, const Edge& b) {
	return a.w < b.w;
}

void iSegment_graph(int num_vertices, int num_edges, Edge*& edges, float c, Universe *u)
{
	Edge* pEdge = edges, *edgesEnd = pEdge + num_edges;
	// sort edges by weight
	std::sort(pEdge, edgesEnd);
	//thrustsort(pEdge,edgesEnd);

	// init thresholds
	float *threshold = new float[num_vertices];
	int i;
	float *pThresh = threshold;
	for (i = 0; i < num_vertices; i++)
		*pThresh++ = THRESHOLD(1, c);

	// for each edge, in non-decreasing weight order...
	while (pEdge != edgesEnd)
	{
		//if(pEdge->valid) {
		// components conected by this edge
		int a = u->find(pEdge->a);
		int b = u->find(pEdge->b);
		if (a != b /*&& a >= 0 && b>= 0 && a < num_vertices && b < num_vertices*/) {
			if ((pEdge->w <= threshold[a]) &&
				(pEdge->w <= threshold[b])) {
				u->join(a, b);
				a = u->find(a);
				if (a < num_vertices && a >= 0)
					threshold[a] = pEdge->w + THRESHOLD(u->size(a), c);
				else
					printf("a is %d, which is out of bounds\n", a);
			}
		}
		//}
		pEdge++;
	}

	// free up
	delete threshold;
}

inline void iJoin_graph(Edge *&edges, int num_edges, int min_size, Universe *u) {
	Edge *pEdge = edges, *edgesEnd = edges + num_edges;
	while (pEdge != edgesEnd)
	{
		int a = u->find(pEdge->a);
		int b = u->find(pEdge->b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
		{
			u->join(a, b);
		}
		pEdge++;
	}
}

void random_rgb(cv::Vec3b &c)
{
	c[0] = rand() % 255 + 1;
	c[1] = rand() % 255 + 1;
	c[2] = rand() % 255 + 1;
}

int FHGraphSegment(
	std::vector<cv::Mat> &in,
	const float sigma,
	const float c,
	const int min_size,
	cv::Mat &out,
	cv::Mat &out_color,
	const int segStartNumber = 0)
{

	int i, size = in[0].rows * in[0].cols;
	Universe u(size);
	Edge* edges = NULL;
	int num_edges;

	iBuildGraph(in, sigma, edges, &num_edges);
	if (edges == NULL || num_edges == 0) {
		printf("Error, graph has no edges\n");
		return 0;
	}
	iSegment_graph(size, num_edges, edges, c, &u);
	iJoin_graph(edges, num_edges, min_size, &u);

	free(edges);

	int numSegs = u.num_sets();
	cv::Vec3b *m_colors = (cv::Vec3b *)malloc(numSegs*sizeof(cv::Vec3b));
	cv::Vec3b *pColor = m_colors;
	for (i = 0; i < numSegs; i++)
	{
		cv::Vec3b color;
		random_rgb(color);
		*pColor++ = color;
	}

	out = cv::Mat(in[0].rows, in[0].cols, CV_32SC1);
	out_color = cv::Mat(in[0].rows, in[0].cols, CV_8UC3);
	cv::Mat_<int>::iterator pO = out.begin<int>();
	cv::Mat_<cv::Vec3b>::iterator pPseudo = out_color.begin<cv::Vec3b>();
	i = 0;
	//We know there are u.num_sets() segments and size possible ids, and we want those ordered starting from 0.
	int *m_ids = (int*)malloc(size*sizeof(int));
	std::fill_n(m_ids, size, -1);
	int currSeg = segStartNumber;
	//std::map<int, int> segmentIdMap;
	while (pO != out.end<int>()) {
		int uId = u.find(i);
		/*if (segmentIdMap.find(uId) == segmentIdMap.end()) {
			segmentIdMap[uId] = currSeg;
			++currSeg;
		}
		*pO = segmentIdMap[uId];*/
		if (m_ids[uId] == -1) {
			m_ids[uId] = currSeg;
			++currSeg;
		}
		*pO = m_ids[uId];
		*pPseudo = m_colors[*pO - segStartNumber];
		++pO; ++i; ++pPseudo;
	}
	free(m_colors);
	u.elts.clear();
	return numSegs;
}

/* END FH */
/* -----------------------------------------------------------------------------------------*/

int segmentation(std::vector<cv::Mat> &in, cv::Mat &out, const int s, const int m, const int iter, const int segStartNumber = 0) {
	/*SLIC slic(in.cols, in.rows);
	slic.ComputeSuperPixels(in, cv::Point(0, 0), s, m, iter);
	slic.DrawResults(out);*/
	cv::Mat tmp;
	int numSegs = FHGraphSegment(in, 0.5f, 200, 200, out, tmp, segStartNumber);
	return numSegs;
}

namespace caffe {

template <typename Dtype>
void SegmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  SegmentationParameter seg_param = this->layer_param_.seg_param();
  height_ = seg_param.data_height();
  width_ = seg_param.data_width();
  //data_height_ = seg_param.data_height();
  //data_width_ = seg_param.data_width();
  seg_parameter_ = seg_param.seg_parameter();
}

template <typename Dtype>
void SegmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";

  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SegmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /*const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* point_data = bottom[1]->cpu_data();
  const Dtype* ground_truth_point_data = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const int num_point = bottom[1]->shape(1) / 2;
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  for (int n = 0; n < num; n++) {
    for (int i = 0; i < num_point; i++) {
      Dtype center_x = (point_data[n * bottom[1]->shape(1) + i * 2] / data_width_ + 0.5)  * bottom[0]->width();
      Dtype center_y = (point_data[n * bottom[1]->shape(1) + i * 2 + 1] / data_height_ + 0.5) * bottom[0]->height();
      int x0 = floor(center_x - width_ / 2);
      int y0 = floor(center_y - height_ / 2);
      if (top.size() == 3) {
        top[2]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2] = (Dtype)(x0 + (Dtype)width_ / 2 + 0.5) / (Dtype)bottom[0]->width() * data_width_;
        top[2]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2 + 1] = (Dtype)(y0 + (Dtype)height_ / 2 + 0.5) / (Dtype)bottom[0]->height() * data_height_;
        top[1]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2] = 
          (ground_truth_point_data[n * bottom[1]->shape(1) + i * 2] + data_width_ / 2) - top[2]->cpu_data()[n * bottom[1]->shape(1) + i * 2];
        top[1]->mutable_cpu_data()[n * bottom[1]->shape(1) + i * 2 + 1] = 
          (ground_truth_point_data[n * bottom[1]->shape(1) + i * 2 + 1] + data_height_ / 2) - top[2]->cpu_data()[n * bottom[1]->shape(1) + i * 2 + 1];

      }

      for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height_; h++) {
          for (int w = 0; w < width_; w++) {
            if (y0 + h >= 0 && y0 + h <= bottom[0]->height() - 1
                && x0 + w >= 0 && x0 + w <= bottom[0]->width() - 1) {
              if (as_dim_ == 0) {
                top_data[top[0]->offset(num * i + n, c, h, w)] = bottom_data[bottom[0]->offset(n, c, y0 + h, x0 + w)];
              }
              else {
                top_data[top[0]->offset(n, channels * i + c, h, w)] = bottom_data[bottom[0]->offset(n, c, y0 + h, x0 + w)];
              }
            }
            else {
              if (as_dim_ == 0) {
                top_data[top[0]->offset(num * i + n, c, h, w)] = 0;
              }
              else {
                top_data[top[0]->offset(n, channels * i + c, h, w)] = 0;
              }
            }
          }
        }
      }
    }
  }*/
	vector<int> shape = bottom[0]->shape();
	int num = shape[0] * shape[1] * shape[2] * shape[3];
	std::vector<cv::Mat> channels = { cv::Mat(), cv::Mat(), cv::Mat() };
	cv::Mat img;
	int segId = 0;
	for (int i = 0; i < shape[0]; i++) {
		channels[0] = cv::Mat(shape[2], shape[3], CV_32FC1, (void*)(bottom[0]->cpu_data() + bottom[0]->offset(i, 0, 0, 0)));
		channels[1] = cv::Mat(shape[2], shape[3], CV_32FC1, (void*)(bottom[0]->cpu_data() + bottom[0]->offset(i, 1, 0, 0)));
		channels[2] = cv::Mat(shape[2], shape[3], CV_32FC1, (void*)(bottom[0]->cpu_data() + bottom[0]->offset(i, 2, 0, 0)));
		cv::merge(channels, img);
		//img.convertTo(img, CV_8UC3);
		cv::Mat seg;
		int numSegs = segmentation(channels, seg, 10, 20, 3, segId);
		segId += numSegs;
	}
	//cv::Mat tmp = DecodeDatumToCVMat(*(const caffe::Datum*)((bottom[0]->cpu_data())), true);
	caffe_copy(num, bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

#ifdef CPU_ONLY
STUB_GPU(SegmentationLayer);
#endif

INSTANTIATE_CLASS(SegmentationLayer);
REGISTER_LAYER_CLASS(Segmentation);

}  // namespace caffe
