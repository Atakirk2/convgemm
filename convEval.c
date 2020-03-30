#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemmConv.h"
#include "convCommon.h"


void maxMatrixSizes(int** model,const int nL, const int maxBatch, int* maxSizeF, int* maxSizeIn, int* maxSizeOut, int* maxSizeAux)
{
    int l,
    sizeF,sizeIn, sizeOut, sizeAux, ho, wo;
    
    *maxSizeF = 0; *maxSizeIn = 0; *maxSizeOut = 0;
    
    for(l=0; l < nL; l++)
    {
		//sizeF = kn* kh *kw*c  
        sizeF = model[l][6] * model[l][4] * model[l][5] * model[l][3];
		//sizeIn = h* w* c * b
        sizeIn = model[l][1] * model[l][2] * model[l][3] * maxBatch;
		
		//ho = floor((h - kh + 2 * pad) / stride + 1);
		ho = floor((model[l][1] - model[l][4] + 2 * model[l][8]) / model[l][7] + 1);
        //wo = floor((w - kw + 2 * pad) / stride + 1);
        wo = floor((model[l][2] - model[l][5] + 2 * model[l][8]) / model[l][7] + 1);
		//sizeOut = ho * wo *kn * b
        sizeOut = ho * wo * model[l][6] * maxBatch;
		//sizeAux = c*kh*kw *ho*wo*b
		sizeAux = model[l][3] * model[l][4] * model[l][5] * ho * wo * maxBatch;
        *maxSizeF = sizeF > *maxSizeF? sizeF: *maxSizeF;
        *maxSizeIn = sizeIn > *maxSizeIn? sizeIn: *maxSizeIn;
        *maxSizeOut = sizeOut  > *maxSizeOut? sizeOut: *maxSizeOut;
		*maxSizeAux = sizeAux  > *maxSizeAux? sizeAux: *maxSizeAux;
    }
    
}

double ** evalNet(int** model, const int nL, const int minBatch, const int maxBatch, const int stepBatch, const int repe)
{
    double **times, tIni, *tConv, *tIm2Col, *tIm2ColGemm, *tImp; //Timing vectors
    double sum;
    int j,r,l, //loop indexes
     	maxSizeF, maxSizeIn, maxSizeOut, maxSizeAux;
    int numBatch;
	int h, w, c, b, kh, kw, kn, stride, pad, ho, wo;//convolution parameters
    float *F, *In, *Out, *Ac_pack, *Bc_pack, *Aux;
	float ONE=1, ZERO=0;
	
	printf("Starting evaluation\n");
    
    //Allocating matrices
    maxMatrixSizes(model, nL, maxBatch, &maxSizeF, &maxSizeIn,&maxSizeOut, &maxSizeAux);

    F = (float*) malloc(maxSizeF * sizeof(float));
    In = (float*) malloc(maxSizeIn * sizeof(float));
    Out = (float*) malloc(maxSizeOut * sizeof(float));
    if(F == NULL || In == NULL || Out == NULL)
    {
        perror("Error allocating matrices:");
        exit(2);
    }
    //auxiliar matrices
    Ac_pack = (float*) aligned_alloc(4096,BLOCK_MC*BLOCK_KC*sizeof(float));
    Bc_pack = (float*) aligned_alloc(4096,BLOCK_KC*BLOCK_NC*sizeof(float));
    Aux = (float*) malloc(maxSizeAux * sizeof(float));
	
	
    bli_srandv(maxSizeF,F,1);
    bli_srandv(maxSizeIn,In,1);

    
    numBatch = (maxBatch -minBatch)/stepBatch+1;
    //Allocating timing structure
	times = (double **) malloc (4 * sizeof(double*));
    tConv = (double *) calloc ((nL+2)*numBatch,sizeof(double));
	tIm2Col = (double *) calloc ((nL+2)*numBatch,sizeof(double));
	tIm2ColGemm= (double *) calloc ((nL+2)*numBatch,sizeof(double));
	tImp= (double *) calloc ((nL+2)*numBatch,sizeof(double));
    if(tConv == NULL || tIm2Col == NULL || tIm2ColGemm == NULL || tImp == NULL)
    {
        perror("Error allocating timing structures:");
        exit(2);
    }
    times[0] = tConv;
	times[1] = tIm2Col;
	times[2] = tIm2ColGemm;
	times[3] = tImp;

    
    
    for(b=minBatch,j=0;b<=maxBatch; b+=stepBatch,j++)
    {
        printf("Evaluation with batch=%d\n",b);
        for(r = 0; r < repe; r++)
        {
                for(l=0; l < nL; l++)
                {
					h = model[l][1]; w = model[l][2]; c = model[l][3];
					kh = model[l][4]; kw = model[l][5]; kn = model[l][6];
					stride = model[l][7]; pad = model[l][8];
					ho = floor((h - kh + 2 * pad) / stride + 1);
					wo = floor((w - kw + 2 * pad) / stride + 1);

					//Timing naive convolution
					tIni = bli_clock();
					convolutionNaive(ho,wo,c,b,In,kh,kw,kn,F,Out, stride);
					tConv[l+j*(nL+2)] += bli_clock() - tIni;

					//Timing im2col +gemm
					tIni = bli_clock();
					im2Col (ho,wo,c,b,In,kh,kw, stride,Aux);
					tIm2Col[l+j*(nL+2)] += bli_clock() -tIni;
					//bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,kn,ho*wo*b,kh*kw*c,&ONE,F,1,kn,Aux,1,kh*kw*c,&ZERO,Out,1,kn);
					sgemm_cust(kn,ho*wo*b,kh*kw*c,1,F,kn,Aux,kh*kw*c,0,Out,kn,Ac_pack,Bc_pack);
					tIm2ColGemm[l+j*(nL+2)] += bli_clock() -tIni;
					
					
					
					//Timing implicint gemm
					tIni = bli_clock();
					sgemm_conv(kh,kw,c,kn,1,F, ho,wo,b, stride, In, 0,Out,Ac_pack,Bc_pack);
					tImp[l+j*(nL+2)] += bli_clock() -tIni;
                }
        }
    
		//Timing averaging and statistics
        for(l=0; l < nL;l++)
		{
            tConv[l+j*(nL+2)] /= repe;
			tIm2Col[l+j*(nL+2)] /= repe;
			tIm2ColGemm[l+j*(nL+2)] /= repe;
			tImp[l+j*(nL+2)] /= repe;
		}
		
        bli_dasumv(nL,&tConv[j*(nL+2)],1,&sum);
        tConv[nL+j*(nL+2)] += sum;
        tConv[nL+1+j*(nL+2)]+= sum/nL;
		
		bli_dasumv(nL,&tIm2Col[j*(nL+2)],1,&sum);
        tIm2Col[nL+j*(nL+2)] += sum;
        tIm2Col[nL+1+j*(nL+2)]+= sum/nL;
		
		bli_dasumv(nL,&tIm2ColGemm[j*(nL+2)],1,&sum);
        tIm2ColGemm[nL+j*(nL+2)] += sum;
        tIm2ColGemm[nL+1+j*(nL+2)]+= sum/nL;
		
		bli_dasumv(nL,&tImp[j*(nL+2)],1,&sum);
        tImp[nL+j*(nL+2)] += sum;
        tImp[nL+1+j*(nL+2)]+= sum/nL;
    }
    
    free(F);
    free(In);
    free(Out);
    
    return times;
}

int loadModel(const char* modelName, int *** modelPtr)
{
    int nL, l = 0, 
		filled;
    FILE* in;
    int **model;
	
        
    in = fopen(modelName,"r");
    if(in==NULL)
    {
        perror("Error opening model file:");
        exit(1);
    }
    
    filled = fscanf(in,"%*6c%d\n",&nL); //read number of layers
	if(filled !=1)
	{
		        perror("Error reading model file:");
				exit(1);
	}
	
    model = (int**) malloc(nL*sizeof(int*));//Alloc model structure
    
    filled = fscanf(in,"%*[^\n]\n");//skip descriptor line
    for(l=0; l < nL; l++)
    {
        model[l] = (int*) malloc(9 * sizeof(int)); //Alloc layer product dimensions
        //model = [id,h,w,c,kh,kw,kn,stride,pad]
		filled = fscanf(in,"%d;%d;%d;%d;%d;%d;%d;%d;%d#\n",&model[l][0],&model[l][1],&model[l][2],&model[l][3],&model[l][4],&model[l][5],&model[l][6],&model[l][7],&model[l][8]);
		if(filled !=9)
		{
			perror("Error reading layer:");
			exit(1);
		}
		//printf("id=%d,h=%d,w=%d,c=%d,kh=%d,kw=%d,kn=%d,stride=%d,pad=%d\n",model[l][0],model[l][1],model[l][2],model[l][3],model[l][4],model[l][5],model[l][6],model[l][7],model[l][8]);

    }
    
    fclose(in);
    
    *modelPtr = model;
    
    printf("Model %s loaded[%d layers]\n",modelName,nL);
    
    return nL;
}

void genOutput(const int nL, int ** model, double** perfMeasures,const int minBatch, const int maxBatch, const int stepBatch,const int repe)
{
    int i, l,j,b;
	int id, h,w,c,kh,kw,kn,stride,pad,ho,wo;
	double gflop, gflopTotal;
	
	
    printf("Evaluation results (averaged results of %d repetitions):\n",repe);
    printf("Execution time batch x layer\n");
    for(b=minBatch,j=0;b<=maxBatch; b+=stepBatch,j++)
    {
        printf("Batch=%d \n",b);
		gflopTotal=0;
		printf("layer \t Naive    \t im2col    \t gemm    \t flops    \t im2colgemm \t implicitGemm  \t flops\n");
        for (l = 0;l < nL; l++)
		{
				id = model[l][0];
				h = model[l][1]; w = model[l][2]; c = model[l][3];
				kh = model[l][4]; kw = model[l][5]; kn = model[l][6];
				stride = model[l][7]; pad = model[l][8];
				ho = floor((h - kh + 2 * pad) / stride + 1);
				wo = floor((w - kw + 2 * pad) / stride + 1);
				
				gflop = ( 2.0 * kn*ho*wo*b*kh*kw*c ) /  1.0e9 ;
				gflopTotal+=gflop;
                
				printf("%d    \t %.4g    \t %.4g    \t %.4g    \t %.5g    \t %.4g    \t %.4g    \t %.5g\n",
				       id,perfMeasures[0][l+j*(nL+2)],perfMeasures[1][l+j*(nL+2)],
					   perfMeasures[2][l+j*(nL+2)]-perfMeasures[1][l+j*(nL+2)],
					   gflop / perfMeasures[2][l+j*(nL+2)],perfMeasures[2][l+j*(nL+2)],
					   perfMeasures[3][l+j*(nL+2)], gflop/perfMeasures[3][l+j*(nL+2)]);
			
		}
		
		printf("Tot \t %.4g    \t %.4g    \t %.4g    \t %.5g    \t %.4g    \t %.4g    \t %.5g\n",
				       perfMeasures[0][l+j*(nL+2)],perfMeasures[1][l+j*(nL+2)],
					   perfMeasures[2][l+j*(nL+2)]-perfMeasures[1][l+j*(nL+2)],
					   gflopTotal / perfMeasures[2][l+j*(nL+2)],perfMeasures[2][l+j*(nL+2)],
					   perfMeasures[3][l+j*(nL+2)], gflopTotal /perfMeasures[3][l+j*(nL+2)]);
		l++;
		printf("Avg \t %.4g    \t %.4g    \t %.4g    \t %.5g    \t %.4g    \t %.4g    \t %.5g\n",
				       perfMeasures[0][l+j*(nL+2)],perfMeasures[1][l+j*(nL+2)],
					   perfMeasures[2][l+j*(nL+2)]-perfMeasures[1][l+j*(nL+2)],
					   (gflopTotal/nL) / perfMeasures[2][l+j*(nL+2)],perfMeasures[2][l+j*(nL+2)],
					   perfMeasures[3][l+j*(nL+2)], (gflopTotal/nL) /perfMeasures[3][l+j*(nL+2)]);
    }
    
}

int main( int argc, char** argv )
{
    char* modelName;
    
    int minBatch,
        maxBatch,
        stepBatch,
        repe,
        nL;
    
    int ** model;
    
    double** perfMeasures;
    
    if (argc != 6)
    {
        printf("Performs an evaluation of the convolution layers involved in a DNN inference.\n");
        printf("\tmodel: file containing the network product structure.\n");
        printf("\tminBatch: minimum batch to test the network .\n");
        printf("\tmaxBatch: maximum batch to test the network.\n");
        printf("\tstepBatch: step for the batch performance evaluation.\n");
        printf("\trepe: number of repetitions of the test.\n");
        printf("Usage: %s <model> <minBatch> <maxBatch> <stepBatch> <repe>\n", argv[0]);
        return -1;
    }

    modelName = argv[1];
    minBatch =atoi(argv[2]);  
    maxBatch =atoi(argv[3]);
    stepBatch = atoi(argv[4]);
    repe  =atoi(argv[5]);

    nL = loadModel(modelName, &model);
    
    perfMeasures = evalNet(model, nL, minBatch, maxBatch, stepBatch, repe);
    
    genOutput(nL, model, perfMeasures, minBatch, maxBatch, stepBatch, repe);

    
    free(perfMeasures);
    free(model);
    
}
