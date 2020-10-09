

#include "performanalysis.h"

/// *** n = should be power of two always *** /// 
// However slight modification can be done to support any n 
 
void time_shared_cached_shared_tile_splitting(const int n, const int num_sys, float *time1)
{	
	int tile_size = 512;							// chosen tile size
	int num_tile = ceil((float)n/tile_size);		// number of tiles
	int gen_sys_size = 1024;						// size of the generated system size
	int num_gen_sys = ceil((float)n/gen_sys_size);	// number of generated systems from one large system
	int num_split = log(num_gen_sys)/log(2.0);		// number of splitting required for the large system
	int cached_area = num_gen_sys;					// required length of cached area
	int buffer_size = tile_size + cached_area;		
	
// allocating device memory for batch of pentadiagobal systems
	double* d_a;  checkCuda(cudaMalloc((void**) &d_a, n*num_sys*sizeof(double)));
	double* d_b;  checkCuda(cudaMalloc((void**) &d_b, n*num_sys*sizeof(double)));
	double* d_c;  checkCuda(cudaMalloc((void**) &d_c, n*num_sys*sizeof(double)));
	double* d_d;  checkCuda(cudaMalloc((void**) &d_d, n*num_sys*sizeof(double)));
	double* d_e;  checkCuda(cudaMalloc((void**) &d_e, n*num_sys*sizeof(double)));
	double* d_y;  checkCuda(cudaMalloc((void**) &d_y, n*num_sys*sizeof(double)));
		
	checkCuda(cudaDeviceSynchronize());
	
// launch the 1st kernel to initialize the batch of pentadiagobal systems
	initialize<<<num_sys*num_tile, tile_size>>>(d_a, d_b, d_c, d_d, d_e, d_y, num_tile, tile_size, n);
    checkCuda( cudaPeekAtLastError() );		checkCuda(cudaDeviceSynchronize());
		      
    int k1 = ceil((float)log(gen_sys_size)/log(2)); // number of cyclic reduction required to solve the generated systems in the 2nd stage
    int k2 = ceil((float)log(n)/log(2));			// number of cyclic reduction required to solve the whole system when n<=1024
    
// time measurement for PCR 
	cudaEvent_t start1, stop1;      
	cudaEventCreate(&start1);	cudaEventCreate(&stop1);      	

 // launching the 2nd kernel to execute PCR on GPU
 	if(n<=1024)
 	{
 		cudaEventRecord(start1, 0);
      	pcr_penta<<<num_sys, n, 6*n*sizeof(double)>>>(d_a, d_b, d_c, d_d, d_e, d_y, k2, n);
		cudaEventRecord(stop1, 0);	checkCuda( cudaPeekAtLastError() );
		cudaEventSynchronize(stop1);       
		cudaEventElapsedTime(time1, start1, stop1);
		checkCuda(cudaDeviceSynchronize());		
 	 } 	
 	else
 	{	
 		cudaEventRecord(start1, 0);
 		
	//splitting kenel
		shared_cached_shared_tile_splitting<<<num_sys, tile_size, 6*buffer_size*sizeof(double)>>>(d_a, 
																								  d_b, 
																								  d_c, 
																								  d_d, 
																								  d_e,
																								  d_y, 
																								  cached_area, 
																								  num_split, 
																								  num_tile,  
																								  buffer_size,  
																								  tile_size, 
																								  n);
	// kenrel for the 2nd stage where all the systems are solved
    	pcr_penta_splitting_2nd_stage<<<num_sys*num_gen_sys, gen_sys_size, 6*(gen_sys_size)*sizeof(double)>>>(d_a, 
    																						   				  d_b, 
    																						   				  d_c, 
    																						   				  d_d, 
    																						   				  d_e, 
    																						   				  d_y, 
    																						   				  k1, 
    																						   				  num_gen_sys, 
    																						   				  gen_sys_size, 
    																						   				  n);      
		cudaEventRecord(stop1, 0);	checkCuda( cudaPeekAtLastError() );
		cudaEventSynchronize(stop1);       
		cudaEventElapsedTime(time1, start1, stop1);
		checkCuda(cudaDeviceSynchronize());		
 	}
 	cudaFree(d_a);   cudaFree(d_b);   cudaFree(d_c);   cudaFree(d_d);   cudaFree(d_e);   cudaFree(d_y);
 }	


 
void time_global_cached_shared_tile_splitting(const int n, const int num_sys, float *time2)
{
	int tile_size = 512;							// chosen tile size
	int num_tile = ceil((float)n/tile_size);		// number of tiles
	int gen_sys_size = 1024;						// size of the generated system size
	int num_gen_sys = ceil((float)n/gen_sys_size);	// number of generated systems from one large system
	int num_split = log(num_gen_sys)/log(2.0);		// number of splitting required for the large system

// allocating device memory for batch of pentadiagobal systems
	double* dd_a;  checkCuda(cudaMalloc((void**) &dd_a, n*num_sys*sizeof(double)));
	double* dd_b;  checkCuda(cudaMalloc((void**) &dd_b, n*num_sys*sizeof(double)));
	double* dd_c;  checkCuda(cudaMalloc((void**) &dd_c, n*num_sys*sizeof(double)));
	double* dd_d;  checkCuda(cudaMalloc((void**) &dd_d, n*num_sys*sizeof(double)));
	double* dd_e;  checkCuda(cudaMalloc((void**) &dd_e, n*num_sys*sizeof(double)));
	double* dd_y;  checkCuda(cudaMalloc((void**) &dd_y, n*num_sys*sizeof(double)));
	
	checkCuda(cudaDeviceSynchronize());
	
// launch the 1st kernel to initialize the batch of pentadiagobal systems
	initialize<<<num_sys*num_tile, tile_size>>>(dd_a, dd_b, dd_c, dd_d, dd_e, dd_y, num_tile, tile_size, n);
    checkCuda( cudaPeekAtLastError() );		checkCuda(cudaDeviceSynchronize());
		      
	int k1 = ceil((float)log(gen_sys_size)/log(2)); // number of cyclic reduction required to solve the generated systems in the 2nd stage
    int k2 = ceil((float)log(n)/log(2));			// number of cyclic reduction required to solve the whole system when n<=1024
    
// time measurement for PCR 
	cudaEvent_t start2, stop2;      
	cudaEventCreate(&start2);	cudaEventCreate(&stop2);    	

 // launching the 2nd kernel to execute PCR on GPU
 	if(n<=1024)
 	{
		cudaEventRecord(start2, 0);
      	pcr_penta<<<num_sys, n, 6*n*sizeof(double)>>>(dd_a, dd_b, dd_c, dd_d, dd_e, dd_y, k2, n);
		cudaEventRecord(stop2, 0);	checkCuda( cudaPeekAtLastError() );
		cudaEventSynchronize(stop2);       
		cudaEventElapsedTime(time2, start2, stop2);
		checkCuda(cudaDeviceSynchronize());		
 	 } 	
 	else
 	{	
		cudaEventRecord(start2, 0);
	//splitting kenel		
		global_cached_shared_tile_splitting<<<num_sys, tile_size, 6*tile_size*sizeof(double)>>>(dd_a, 
																								dd_b, 
																								dd_c, 
																								dd_d, 
																								dd_e, 
																								dd_y, 
																								num_split, 
																								num_tile, 
																								tile_size, 
																								n);
	// kenrel for the 2nd stage where all the systems are solved
    	pcr_penta_splitting_2nd_stage<<<num_sys*num_gen_sys, gen_sys_size, 6*(gen_sys_size)*sizeof(double)>>>(dd_a, 
    																						   				  dd_b, 
    																						   				  dd_c, 
    																						   				  dd_d, 
    																						   				  dd_e, 
    																						   				  dd_y, 
    																						   				  k1, 
    																						   				  num_gen_sys, 
    																						   				  gen_sys_size, 
    																						   				  n);      
		cudaEventRecord(stop2, 0);	checkCuda( cudaPeekAtLastError() );
		cudaEventSynchronize(stop2);       
		cudaEventElapsedTime(time2, start2, stop2);
		checkCuda(cudaDeviceSynchronize());
 	}
 	cudaFree(dd_a);  cudaFree(dd_b);  cudaFree(dd_c);  cudaFree(dd_d);  cudaFree(dd_e);  cudaFree(dd_y);
}
 
 
 void time_global_cached_global_tile_splitting(const int n, const int num_sys, float *time3)
{
	int tile_size = 512;							// chosen tile size
	int num_tile = ceil((float)n/tile_size);		// number of tiles
	int gen_sys_size = 1024;						// size of the generated system size
	int num_gen_sys = ceil((float)n/gen_sys_size);	// number of generated systems from one large system
	int num_split = log(num_gen_sys)/log(2.0);		// number of splitting required for the large system
	
	// allocating device memory for batch of pentadiagobal systems
	double* ddd_a;  checkCuda(cudaMalloc((void**) &ddd_a, n*num_sys*sizeof(double)));
	double* ddd_b;  checkCuda(cudaMalloc((void**) &ddd_b, n*num_sys*sizeof(double)));
	double* ddd_c;  checkCuda(cudaMalloc((void**) &ddd_c, n*num_sys*sizeof(double)));
	double* ddd_d;  checkCuda(cudaMalloc((void**) &ddd_d, n*num_sys*sizeof(double)));
	double* ddd_e;  checkCuda(cudaMalloc((void**) &ddd_e, n*num_sys*sizeof(double)));
	double* ddd_y;  checkCuda(cudaMalloc((void**) &ddd_y, n*num_sys*sizeof(double)));
	
	checkCuda(cudaDeviceSynchronize());
	
	// launch the 1st kernel to initialize the batch of pentadiagobal systems
	initialize<<<num_sys*num_tile, tile_size>>>(ddd_a, ddd_b, ddd_c, ddd_d, ddd_e, ddd_y, num_tile, tile_size, n);
    checkCuda( cudaPeekAtLastError() );		checkCuda(cudaDeviceSynchronize());
		
	int k1 = ceil((float)log(gen_sys_size)/log(2)); // number of cyclic reduction required to solve the generated systems in the 2nd stage
    int k2 = ceil((float)log(n)/log(2));			// number of cyclic reduction required to solve the whole system when n<=1024
    
// time measurement for PCR 
	cudaEvent_t start3, stop3;      
	cudaEventCreate(&start3);	cudaEventCreate(&stop3);      	

 // launching the 2nd kernel to execute PCR on GPU
 	if(n<=1024)
 	{
		cudaEventRecord(start3, 0);
      	pcr_penta<<<num_sys, n, 6*n*sizeof(double)>>>(ddd_a, ddd_b, ddd_c, ddd_d, ddd_e, ddd_y, k2, n);
		cudaEventRecord(stop3, 0);	checkCuda( cudaPeekAtLastError() );
		cudaEventSynchronize(stop3);       
		cudaEventElapsedTime(time3, start3, stop3);			
 	 } 	
 	else
 	{	
 		cudaEventRecord(start3, 0);
 	//splitting kenel	
		global_cached_global_tile_splitting<<<num_sys, tile_size>>>(ddd_a, 
																	ddd_b, 
																	ddd_c, 
																	ddd_d, 
																	ddd_e, 
																	ddd_y, 
																	num_split, 
																	num_tile, 
																	tile_size, 
																	n);
	// kenrel for the 2nd stage where all the systems are solved																
    	pcr_penta_splitting_2nd_stage<<<num_sys*num_gen_sys, gen_sys_size, 6*(gen_sys_size)*sizeof(double)>>>(ddd_a, 
    																						   				  ddd_b, 
    																						   				  ddd_c, 
    																						   				  ddd_d, 
    																						   				  ddd_e, 
    																						   				  ddd_y, 
    																						   				  k1, 
    																						   				  num_gen_sys, 
    																						   				  gen_sys_size, 
    																						   				  n);      
		cudaEventRecord(stop3, 0);	checkCuda( cudaPeekAtLastError() );
		cudaEventSynchronize(stop3);       
		cudaEventElapsedTime(time3, start3, stop3);
		checkCuda(cudaDeviceSynchronize());				
 	}
 	cudaFree(ddd_a); cudaFree(ddd_b); cudaFree(ddd_c); cudaFree(ddd_d); cudaFree(ddd_e); cudaFree(ddd_y);
}
 
 

int main()
{ 	
 	float *time1 = (float *)malloc(sizeof(float));
   	float *time2 = (float *)malloc(sizeof(float));
   	float *time3 = (float *)malloc(sizeof(float));
   	  
    int p = 17;	int q = 10;
    int batchsize[17] = {26, 52, 104, 156, 208, 260, 312, 364, 416, 624, 832, 1040, 1248, 1456, 1664, 1872, 2080};
    int syssize[10] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    
 	FILE *fp; 
	fp = fopen("time_perform_analysis_oddeven.dat", "w"); // write only
// test for files not existing. 
  	if (fp == NULL) 
    	{ printf("Error! Could not open file\n"); 	exit(-1); } 
    fclose(fp);
    	 	   	 	
   	int avg_run = 10;
   			 
 	for(int j = 0; j<p; j++)
 	{
 	 	for(int i = 0; i<q; i++)
 	    { 	    		
 	        float sum1 = 0.0;	float sum2 = 0.0;	float sum3 = 0.0; 		         	
 	        *time1 = 0.0;			*time2 = 0.0;	 	*time3 = 0.0;	
 	        
 	        if(j> 12 && i> 8) // these batch and system sizes lead to out of memory
 	        {	
 	        	fp = fopen("time_perform_analysis_oddeven.dat", "a");
 	        	if (fp == NULL) 
				{ printf("Error! Could not open file\n"); 	exit(-1); }
				fprintf (fp, "%d \t %d \t %f \t %f \t %f\n", batchsize[j], syssize[i], 0.0, 0.0, 0.0);
				fclose(fp);
				continue;
 	        } 	        	
 	        else
 	        {
				for(int k = 0; k<avg_run; k++)
				{						
					time_shared_cached_shared_tile_splitting(syssize[i], batchsize[j], time1);
					checkCuda(cudaDeviceSynchronize());
					checkCuda(cudaDeviceReset());
					
					time_global_cached_shared_tile_splitting(syssize[i], batchsize[j], time2);
					checkCuda(cudaDeviceSynchronize());
					checkCuda(cudaDeviceReset());
					
					time_global_cached_global_tile_splitting(syssize[i], batchsize[j], time3);
					checkCuda(cudaDeviceSynchronize());
					checkCuda(cudaDeviceReset());		
				
					 sum1+=*time1;	sum2+=*time2;	sum3+=*time3;					
				}
			printf(" %d \t %d\n", batchsize[j], syssize[i]);
						
			fp = fopen("time_perform_analysis_oddeven.dat", "a"); 
	// test for files not existing. 
			if (fp == NULL) 
				{ printf("Error! Could not open file\n"); 	exit(-1); }
			fprintf (fp, "%d \t %d \t %f \t %f \t %f\n", batchsize[j], syssize[i], sum1/avg_run, sum2/avg_run, sum3/avg_run);
			fclose(fp);
			}			
 	 	}
 	 }

 	return 0;
}
