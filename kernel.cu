
#include "performanalysis.h"

// this kernel will initialize the pentadiagonal matrix and right hand vector
__global__ void initialize(double *d_a, double *d_b, double *d_c, double *d_d, double *d_e, double *d_y, int nsys, int syssize, int n)
	{
// i is the equation number
	int i = threadIdx.x; 	int j = blockIdx.x;					int sys_id = fmodf(j, nsys); 
	int Lsys_id = j/nsys;	int eqn = sys_id*syssize + i;  		int g_id = Lsys_id*n + eqn;
	if(eqn<n)
	  {
           /*d_a[i] = 1.0;	 d_b[i] = 2.0; 	d_c[i] = 8.0; 
           d_d[i] = 2.0; 	d_e[i] = 1.0; 	d_y[i] = 8.0;
           if (id == 0)			{d_a[i] = 0.0; 	d_b[i] = 1.0;}            
           else if (id == 1) 	d_a[i] = 0.0;
           else if (id == n-2)	d_e[i] = 0.0;           
           else if (id == n-1)	{d_d[i] = 1.0;		d_e[i] = 0.0;}*/
	       
	   d_a[g_id] = 1.0; 	d_b[g_id] = -4.0; 	d_c[g_id] = 6.0; 
	   d_d[g_id] = -4.0; 	d_e[g_id] = 1.0; 	d_y[g_id] = 1.0;           
           if (eqn == 0)   {d_a[g_id] = 0.0; 	d_b[g_id] = 1.0; 	d_c[g_id] = 5.0;}
           if (eqn == 1)    d_a[g_id] = 0.0;	              
           if (eqn == n-2)  d_e[g_id] = 0.0;	              
           if (eqn == n-1) {d_c[g_id] = 5.0; 	d_d[g_id] = 1.0; 	d_e[g_id] = 0.0;}
	  }        
	}  
   
   


// this kernel will Parallely solves the pentadiagonal systems of size <= 1024
__global__ void pcr_penta(double *d_a, 
			  double *d_b, 
			  double *d_c, 
			  double *d_d, 
			  double *d_e, 
			  double* d_y, 
			  int num_split,		// number of splitting  
			   int n)
{ 
	int id = threadIdx.x; 	// equation id
	int j = blockIdx.x;		// system id
	int g_id = j*n + id;	// global equation id
	
// allocating memory in the shared memory for all 5 diags and right hand vector
	extern __shared__ double array [];
		 
	double *a = array;	 				double *b = (double*) &a[n];
    double *c = (double*) &b[n];	 	double *d = (double*) &c[n];
    double *e = (double*) &d[n];	 	double *x = (double*) &e[n];
    
// copying system from global memory to shared memory
	a[id] = d_a[g_id]; 		b[id] = d_b[g_id]; 		c[id] = d_c[g_id]; 
	d[id] = d_d[g_id]; 		e[id] = d_e[g_id]; 		x[id] = d_y[g_id];
	
// waiting for every thread to finish copying
   	 __syncthreads();
     
	int delta = 1;	int p1, p2, q1, q2; 
    double alfa1, alfa2, a1, b1, c1, d1, e1, k1, a2, b2, c2, d2, e2, k2;
    
// PCR-Penta reduction loop
  	for(int l = 0; l< num_split-1; l++)
	{
		// Calculating ids of the required neighbors
		p1 = id - delta;	  p2 = id - 2*delta;	q1 = id + delta;	  q2 = id + 2*delta;

	//Reduction-I            
		if(p2 >= 0)	// if both neighbors required in Reduction-I exist in the system
        {       
            alfa1 = 0.0;		alfa2 = 0.0;
            if((id - 3*delta) >= 0)
                if( b[p2] != 0.0 )	alfa1 = -a[p1]/b[p2];
            if(q1 <= n-1)
                 if( d[id] != 0.0 )	alfa2 = -e[p1]/d[id];

            a1 = alfa1*a[p2];	     					b1 = alfa1*c[p2] + b[p1] + alfa2*a[id];
            c1 = alfa1*d[p2] + c[p1] + alfa2*b[id];		d1 = alfa1*e[p2] + d[p1] + alfa2*c[id];
            e1 = alfa2*e[id]; 							k1 = alfa1*x[p2] + x[p1] + alfa2*x[id];
        }

        if(p1 >= 0 && p2 < 0)	// if only one neighbor required in Reduction-I exist 
        {   					// in the system and other lies outside of the system
         	alfa2 = 0.0;
            if(q1 <= n-1)
              	 if( d[id] != 0.0 ) 	alfa2 = -e[p1]/d[id]; 

            a1 = 0.0;						b1 = 0.0;
            c1 = c[p1] + alfa2*b[id];		d1 = d[p1] + alfa2*c[id];
            e1 = alfa2*e[id]; 				k1 = x[p1] + alfa2*x[id];
         }

	// Reduction-II
        if(q2 <= n-1 )	// if both neighbors required in Reduction-II exist in the system
        { 	
        	alfa1 = 0.0; 		alfa2 = 0.0;

            if(p1 >= 0)
            	if( b[id] != 0.0 )		alfa1 = -a[q1]/b[id];
             if((id + 3*delta) <= n-1)
             	if( d[q2] != 0.0)   	alfa2 = -e[q1]/d[q2];

             a2 = alfa1*a[id];							b2 = alfa1*c[id] + b[q1] + alfa2*a[q2];
             c2 = alfa1*d[id] + c[q1] + alfa2*b[q2];	d2 = alfa1*e[id] + d[q1] + alfa2*c[q2];
             e2 = alfa2*e[q2]; 							k2 = alfa1*x[id] + x[q1] + alfa2*x[q2];
          }
          
		if(q1 <= n-1 && q2 > n-1) 	// if only one neighbor required in Reduction-II exist 
		{							// in the system and other lies outside of the system
            alfa1 = 0.0;
			if(p1 >= 0)
            	if(b[id] != 0.0)   	alfa1 = -a[q1]/b[id];

            a2 = alfa1*a[id];				b2 = alfa1*c[id] + b[q1];
            c2 = alfa1*d[id] + c[q1];		d2 = 0.0;
            e2 = 0.0; 						k2 = alfa1*x[id] + x[q1];
         }

// Reduction-III
         if(p1 >= 0)
         {
          	if( c1 != 0.0 )   	alfa1 = b[id]/c1;
            	a1 = alfa1*a1;	b1 = alfa1*b1;	c1 = alfa1*d1;	d1 = alfa1*e1;	k1 = alfa1*k1;
	     }
         else	{a1 = 0.0;	b1 = 0.0;	c1 = 0.0;	d1 = 0.0;	k1 = 0.0;}

         if(q1 <= n-1)
         {
	     	if( c2 != 0.0 )   	alfa2 = d[id]/c2;
            c2 = alfa2*b2;  b2 = alfa2*a2;	d2 = alfa2*d2; 	e2 = alfa2*e2;	k2 = alfa2*k2;
	      }
	      else	{b2 = 0.0;	c2 = 0.0;	d2 = 0.0;	e2 = 0.0;	k2 = 0.0;}

		// waiting for every threads to complete Reduction-I and -II 
		  __syncthreads();
		  
		  delta*=2;
	      b[id] = b1 - a[id] + b2;		c[id] = c1 - c[id] + c2;
	      d[id] = d1 - e[id] + d2; 		x[id] = k1 - x[id] + k2;
	      a[id] = a1; 					e[id] = e2;
		// waiting for every threads to complete Reduction-III 
    	  __syncthreads();  	   	 
    }
// for the last loop system size is less than or equal to 2, thus efficient to solve them manually
	if(id < n-delta)
	{
		int iid = id+delta;
		double deno = c[id]*c[iid]-b[iid]*d[id];
		d_y[g_id] = (c[iid]*x[id]-d[id]*x[iid])/deno;
		d_y[g_id + delta] = (c[id]*x[iid]-b[iid]*x[id])/deno;
	}
	if(id >= n-delta && id < delta)
  	   d_y[g_id] = x[id]/c[id];
}



__global__ void shared_cached_shared_tile_splitting(double *d_a, 
						    						double *d_b, 
						    						double *d_c, 
						    						double *d_d, 
						    						double *d_e, 
						   	 						double* d_y, 
						    						int chached_area,	// lenght of cached area
						    						int num_split,		// number of splitting  	
						    						int num_tile, 		// number of tiles
						    						int buffer_size, 	// total lenght of chached area + tile size
						    						int tile_size, 		// tile size
						    						int n)
	{
		int i = threadIdx.x; 	int j = blockIdx.x;		int id = i + chached_area;
// allocating memory in the shared memory for all 5 diags and right hand vector
		extern __shared__ double array [];
		 
		double *a = array;	 						double *b = (double*) &a[buffer_size];
    	double *c = (double*) &b[buffer_size];	 	double *d = (double*) &c[buffer_size];
    	double *e = (double*) &d[buffer_size];	 	double *x = (double*) &e[buffer_size];
    	
    	int delta = 1;	 int ddelta = 2;	int p1, p2, q1, q2;
        double alfa1, alfa2, a1, b1, c1, d1, e1, k1, a2, b2, c2, d2, e2, k2;
    	
    	for(int lc = 0; lc<num_split; lc++)	// splitting loop
    	{
    		for(int t = 0; t<num_tile; t++)	// tile loop
    		{	   		
    			int eqn_id = tile_size*t + i;			// equation id
    			int g_id = j*n + eqn_id;				// equation global id
    			int tile_end_id = tile_size*(t+1)-1;	// id of last equation of the current tile
    			   			
    			if(eqn_id < n)	// loading tile data from global to shared memory
    			{			
    		 		a[id] = d_a[g_id]; 	b[id] = d_b[g_id]; 	c[id] = d_c[g_id]; 
    		 		d[id] = d_d[g_id]; 	e[id] = d_e[g_id]; 	x[id] = d_y[g_id];
    		 	}	 		 	
    		 	__syncthreads();
   		 	

				if(eqn_id < n)
    			{
					// Calculating ids of the required neighbors				
            		p1 = id - delta;	  p2 = id - ddelta;		q1 = id + delta;	  q2 = id + ddelta;
					
		//Reduction-I
            		if(eqn_id - ddelta >= 0)	// if both neighbors required in Reduction-I exist in the system
            		{       
               			alfa1 = 0.0;		alfa2 = 0.0;

               			if((eqn_id - 3*delta) >= 0)
                 			if( b[p2] != 0.0 )	alfa1 = -a[p1]/b[p2];

               			if((eqn_id + delta) <= n-1)
                  			if( d[id] != 0.0 )		alfa2 = -e[p1]/d[id];

               			a1 = alfa1*a[p2];	     					b1 = alfa1*c[p2] + b[p1] + alfa2*a[id];
              			c1 = alfa1*d[p2] + c[p1] + alfa2*b[id];		d1 = alfa1*e[p2] + d[p1] + alfa2*c[id];
               			e1 = alfa2*e[id]; 							k1 = alfa1*x[p2] + x[p1] + alfa2*x[id];
             		}

            		if(eqn_id - delta >= 0 && eqn_id - ddelta < 0)	// if only one neighbor required in Reduction-I exist 
        			{   											// in the system and other lies outside of the system
               			alfa2 = 0.0;

               			if((eqn_id + delta) <= n-1)
              	   			if( d[id] != 0.0 ) 		alfa2 = -e[p1]/d[id]; 

               			a1 = 0.0;						b1 = 0.0;
               			c1 = c[p1] + alfa2*b[id];		d1 = d[p1] + alfa2*c[id];
               			e1 = alfa2*e[id]; 				k1 = x[p1] + alfa2*x[id];
                 	}
                 	
		// Reduction-II
            		if(eqn_id + ddelta <= n-1 )	// if both neighbors required in Reduction-II exist in the system
        			{ 	
        				alfa1 = 0.0; 		alfa2 = 0.0;
        			
	        			if(eqn_id + delta <= tile_end_id && eqn_id + ddelta > tile_end_id) // if only one neighbor lies inside the tile
	            		{	
	            			int q2_id = g_id + ddelta;
	            			if((eqn_id - delta) >= 0)
	            				if( b[id] != 0.0 )		alfa1 = -a[q1]/b[id];

	        				if((eqn_id + 3*delta) <= n-1)
	            				if( d_d[q2_id] != 0.0)   	alfa2 = -e[q1]/d_d[q2_id];

	        				a2 = alfa1*a[id];							 	b2 = alfa1*c[id] + b[q1] + alfa2*d_a[q2_id];
	        				c2 = alfa1*d[id] + c[q1] + alfa2*d_b[q2_id];	d2 = alfa1*e[id] + d[q1] + alfa2*d_c[q2_id];
	            			e2 = alfa2*d_e[q2_id]; 						 	k2 = alfa1*x[id] + x[q1] + alfa2*d_y[q2_id];
	                	}                	
	                	else if(eqn_id + delta> tile_end_id) 	// if both neighbors lie outside of the current tile
	            		{	
	            			int q1_id = g_id + delta; 	int q2_id = g_id + ddelta;
	            			if((eqn_id - delta) >= 0)
	            				if( b[id] != 0.0 )		alfa1 = -d_a[q1_id]/b[id];

	        				if((eqn_id + 3*delta) <= n-1)
	            				if( d_d[q2_id] != 0.0)   	alfa2 = -d_e[q1_id]/d_d[q2_id];

	        				a2 = alfa1*a[id];							 		b2 = alfa1*c[id] + d_b[q1_id] + alfa2*d_a[q2_id];
	        				c2 = alfa1*d[id] + d_c[q1_id] + alfa2*d_b[q2_id];	d2 = alfa1*e[id] + d_d[q1_id] + alfa2*d_c[q2_id];
	            			e2 = alfa2*d_e[q2_id]; 						 		k2 = alfa1*x[id] + d_y[q1_id] + alfa2*d_y[q2_id];
	                	}
	                	else	 // if both neighbors lie inside of the current tile
						{
							if((eqn_id - delta) >= 0)
	            				if( b[id] != 0.0 )		alfa1 = -a[q1]/b[id];

		        			if((eqn_id + 3*delta) <= n-1)
		            			if( d[q2] != 0.0)   	alfa2 = -e[q1]/d[q2];

		        			a2 = alfa1*a[id];							b2 = alfa1*c[id] + b[q1] + alfa2*a[q2];
		        			c2 = alfa1*d[id] + c[q1] + alfa2*b[q2];		d2 = alfa1*e[id] + d[q1] + alfa2*c[q2];
		            		e2 = alfa2*e[q2]; 						 	k2 = alfa1*x[id] + x[q1] + alfa2*x[q2];
						}        			
                	}

            		if(eqn_id + delta <= n-1 && eqn_id + ddelta > n-1) 	// if only one neighbor required in Reduction-II exist 
					{													// in the system and other lies outside of the system	
                		alfa1 = 0.0;

              			if((eqn_id - delta) >= 0)
              	   			if(b[id] != 0.0)   alfa1 = -a[q1]/b[id];

              			a2 = alfa1*a[id];				b2 = alfa1*c[id] + b[q1];
              			c2 = alfa1*d[id] + c[q1];		d2 = 0.0;
              			e2 = 0.0; 						k2 = alfa1*x[id] + x[q1];
                  	}

		// Reduction-III
            		if(eqn_id - delta >= 0)
            		{
               			if( c1 != 0.0 )   	alfa1 = b[id]/c1;
               				a1 = alfa1*a1;	b1 = alfa1*b1;	c1 = alfa1*d1; d1 = alfa1*e1; k1 = alfa1*k1;
	             	}
            		else	{a1 = 0.0;	b1 = 0.0;	c1 = 0.0;	d1 = 0.0;	k1 = 0.0;}

            		if(eqn_id + delta <= n-1)
            		{
	       				if( c2 != 0.0 )   	alfa2 = d[id]/c2;
               				c2 = alfa2*b2;  b2 = alfa2*a2;	d2 = alfa2*d2; e2 = alfa2*e2;	k2 =alfa2*k2;
	             	}
	     			else	{b2 = 0.0;	c2 = 0.0;	d2 = 0.0;	e2 = 0.0;	k2 = 0.0;}
	     			
					// reduced tile data are updating in the corresponding global memory
	    			d_b[g_id] =  b1 - a[id] + b2;			d_c[g_id] = c1 - c[id] + c2;
	    			d_d[g_id] =  d1 - e[id] + d2; 			d_y[g_id] = k1 - x[id] + k2;
	    			d_a[g_id] = a1;							d_e[g_id] = e2;  
    			}
    			
    			if(i>= tile_size - ddelta && t< num_tile-1 ) // required data for processing the next tile is cached 		 	
		 		{											 // from tile loading area to chached area
		 			int idd = id - tile_size;
		 			a[idd] = a[id]; 	b[idd] = b[id]; 	c[idd] = c[id]; 
		 			d[idd] = d[id]; 	e[idd] = e[id]; 	x[idd] = x[id];
		 		}   		 		   		 				 	    		 				
    			__syncthreads();
    		}
    		delta*=2;
    		ddelta*=2;
    	}
    }
     
   
   
 
__global__ void global_cached_shared_tile_splitting(double *d_a, 
													double *d_b, 
													double *d_c, 
													double *d_d, 
													double *d_e, 
													double* d_y, 
						    						int num_split,		// number of splitting  	
						    						int num_tile, 		// number of tiles
						    						int tile_size, 		// tile size
													int n)
	{
		int id = threadIdx.x; 	int j = blockIdx.x;
// allocating memory in the shared memory for all 5 diags and right hand vector
		extern __shared__ double array [];
		 
		double *a = array;	 						double *b = (double*) &a[tile_size];
    	double *c = (double*) &b[tile_size];	 	double *d = (double*) &c[tile_size];
    	double *e = (double*) &d[tile_size];	 	double *x = (double*) &e[tile_size];
    	
    	int delta = 1;	 int ddelta = 2;	int p1, p2, q1, q2;
        double alfa1, alfa2, a1, b1, c1, d1, e1, k1, a2, b2, c2, d2, e2, k2;
    	
    	for(int lc = 0; lc<num_split; lc++)	// splitting loop
    	{
    		for(int t = 0; t<num_tile; t++)	// tile loop
    		{	    		
    			int tile_strt_id = tile_size*t;  		// id of first equation of the current tile
    			int tile_end_id = tile_size*(t+1)-1; 	// id of last equation of the current tile
    			int eqn_id = tile_strt_id + id;			// equation id
    			int g_id = j*n + eqn_id;				// equation global id
    			
    			if(eqn_id < n)	// loading tile data from global to shared memory
    			{   						
	    		 	a[id] = d_a[g_id]; 	b[id] = d_b[g_id]; 	c[id] = d_c[g_id]; 
	    		 	d[id] = d_d[g_id]; 	e[id] = d_e[g_id]; 	x[id] = d_y[g_id];
	    		 }

    		 	__syncthreads();
    		 	
				// Calculating ids of the required neighbors
            	p1 = id - delta;	  p2 = id - ddelta;		q1 = id + delta;	  q2 = id + ddelta;
            	
		//Reduction-I
            	if(eqn_id - ddelta >= 0 && eqn_id < n)	// if both neighbors required in Reduction-I exist in the system
            	{       
               		alfa1 = 0.0;		alfa2 = 0.0;
               		               							
	            	if(eqn_id - delta >= tile_strt_id && eqn_id - ddelta < tile_strt_id) // if only one neighbor lies inside the tile
	            	{
	            		int p2_id = g_id-ddelta;
	            	 	if((eqn_id - 3*delta) >= 0)
	                 		if( d_b[p2_id] != 0.0 )	alfa1 = -a[p1]/d_b[p2_id];

	               		if((eqn_id + delta) <= n-1)
	                  		if( d[id] != 0.0 )	alfa2 = -e[p1]/d[id];

	               		a1 = alfa1*d_a[p2_id];	     					b1 = alfa1*d_c[p2_id] + b[p1] + alfa2*a[id];
	              		c1 = alfa1*d_d[p2_id] + c[p1] + alfa2*b[id];	d1 = alfa1*d_e[p2_id] + d[p1] + alfa2*c[id];
	               		e1 = alfa2*e[id]; 								k1 = alfa1*d_y[p2_id] + x[p1] + alfa2*x[id];
	            	}	            	
	            	else if(eqn_id - delta < tile_strt_id)	// if both neighbor lies outside the current tile
	            	{
	            		int p1_id = g_id-delta;		int p2_id = g_id-ddelta;
	            	 	if((eqn_id - 3*delta) >= 0)
	                 		if( d_b[p2_id] != 0.0 )	alfa1 = -d_a[p1_id]/d_b[p2_id];

	               		if((eqn_id + delta) <= n-1)
	                  		if( d[id] != 0.0 )	alfa2 = -d_e[p1_id]/d[id];

	               		a1 = alfa1*d_a[p2_id];	     						b1 = alfa1*d_c[p2_id] + d_b[p1_id] + alfa2*a[id];
	              		c1 = alfa1*d_d[p2_id] + d_c[p1_id] + alfa2*b[id];	d1 = alfa1*d_e[p2_id] + d_d[p1_id] + alfa2*c[id];
	               		e1 = alfa2*e[id]; 									k1 = alfa1*d_y[p2_id] + d_y[p1_id] + alfa2*x[id];
	            	}
	            	else	// if both neighbor lies inside the current tile
	            	{
	            		if((eqn_id - 3*delta) >= 0)
                 		if( b[p2] != 0.0 )	alfa1 = -a[p1]/b[p2];

	               		if((eqn_id + delta) <= n-1)
	                  		if( d[id] != 0.0 )	alfa2 = -e[p1]/d[id];

	               		a1 = alfa1*a[p2];	     					b1 = alfa1*c[p2] + b[p1] + alfa2*a[id];
	              		c1 = alfa1*d[p2] + c[p1] + alfa2*b[id];		d1 = alfa1*e[p2] + d[p1] + alfa2*c[id];
	               		e1 = alfa2*e[id]; 							k1 = alfa1*x[p2] + x[p1] + alfa2*x[id];
	            	}							
             	}

            	if((eqn_id - delta >= 0 && eqn_id - ddelta < 0) && eqn_id < n)	// if only one neighbor required in Reduction-I exist 
        		{   															// in the system and other lies outside of the system
               		alfa2 = 0.0;

               		if((eqn_id + delta) <= n-1)
              	   		if( d[id] != 0.0 ) 	alfa2 = -e[p1]/d[id]; 

               		a1 = 0.0;						b1 = 0.0;
               		c1 = c[p1] + alfa2*b[id];		d1 = d[p1] + alfa2*c[id];
               		e1 = alfa2*e[id]; 				k1 = x[p1] + alfa2*x[id]; 
                 }
				
				__syncthreads();
				
				if(t > 0)		// updating the reduce data of the previous tile in the current tile
				{
					int pre_id = g_id - tile_size;
				 	d_a[pre_id] = a2;		d_b[pre_id] = b2;		d_c[pre_id] = c2;
				 	d_d[pre_id] = d2; 		d_y[pre_id] = k2;		d_e[pre_id] = e2;
				}
								
		//Reduction-II
        		if(eqn_id + ddelta <= n-1 ) // if both neighbors required in Reduction-II exist in the system
        		{ 	
        			alfa1 = 0.0; 		alfa2 = 0.0;
        			
        			if(eqn_id + delta <= tile_end_id && eqn_id + ddelta > tile_end_id) // if only one neighbor lies inside the tile
            		{	
            			int q2_id = g_id + ddelta;
            			if((eqn_id - delta) >= 0)
            				if( b[id] != 0.0 )		alfa1 = -a[q1]/b[id];

        				if((eqn_id + 3*delta) <= n-1)
            				if( d_d[q2_id] != 0.0)   	alfa2 = -e[q1]/d_d[q2_id];

        				a2 = alfa1*a[id];							 	b2 = alfa1*c[id] + b[q1] + alfa2*d_a[q2_id];
        				c2 = alfa1*d[id] + c[q1] + alfa2*d_b[q2_id];	d2 = alfa1*e[id] + d[q1] + alfa2*d_c[q2_id];
            			e2 = alfa2*d_e[q2_id]; 						 	k2 = alfa1*x[id] + x[q1] + alfa2*d_y[q2_id];
                	}                	
                	else if(eqn_id + delta> tile_end_id) 	// if both neighbors lie outside of the current tile
            		{	
            			int q1_id = g_id + delta; 	int q2_id = g_id + ddelta;
            			if((eqn_id - delta) >= 0)
            				if( b[id] != 0.0 )		alfa1 = -d_a[q1_id]/b[id];

        				if((eqn_id + 3*delta) <= n-1)
            				if( d_d[q2_id] != 0.0)   	alfa2 = -d_e[q1_id]/d_d[q2_id];

        				a2 = alfa1*a[id];							 		b2 = alfa1*c[id] + d_b[q1_id] + alfa2*d_a[q2_id];
        				c2 = alfa1*d[id] + d_c[q1_id] + alfa2*d_b[q2_id];	d2 = alfa1*e[id] + d_d[q1_id] + alfa2*d_c[q2_id];
            			e2 = alfa2*d_e[q2_id]; 						 		k2 = alfa1*x[id] + d_y[q1_id] + alfa2*d_y[q2_id];
                	}
                	else	// if both neighbors lie inside of the current tile
					{
						if((eqn_id - delta) >= 0)
            				if( b[id] != 0.0 )		alfa1 = -a[q1]/b[id];

	        			if((eqn_id + 3*delta) <= n-1)
	            			if( d[q2] != 0.0)   	alfa2 = -e[q1]/d[q2];

	        			a2 = alfa1*a[id];							b2 = alfa1*c[id] + b[q1] + alfa2*a[q2];
	        			c2 = alfa1*d[id] + c[q1] + alfa2*b[q2];		d2 = alfa1*e[id] + d[q1] + alfa2*c[q2];
	            		e2 = alfa2*e[q2]; 						 	k2 = alfa1*x[id] + x[q1] + alfa2*x[q2];
					}        			
                }

				if(eqn_id + delta <= n-1 && eqn_id + ddelta > n-1) 	// if only one neighbor required in Reduction-II exist 
				{													// in the system and other lies outside of the system	
					alfa1 = 0.0;

					if((eqn_id - delta) >= 0)
			   			if(b[id] != 0.0)   	alfa1 = -a[q1]/b[id];

					a2 = alfa1*a[id];				b2 = alfa1*c[id] + b[q1];
					c2 = alfa1*d[id] + c[q1];		d2 = 0.0;
					e2 = 0.0; 						k2 = alfa1*x[id] + x[q1];
			    }

		// Reduction-III
        		if(eqn_id - delta >= 0 && eqn_id < n)
        		{
        			if( c1 != 0.0 )   	alfa1 = b[id]/c1;
        				a1 = alfa1*a1;	b1 = alfa1*b1;	c1 = alfa1*d1; d1 = alfa1*e1; k1 = alfa1*k1;
             	}
        		else	{a1 = 0.0;	b1 = 0.0;	c1 = 0.0;	d1 = 0.0;	k1 = 0.0;}

        		if(eqn_id + delta <= n-1)
        		{
       				if( c2 != 0.0 )   	alfa2 = d[id]/c2;
            			c2 = alfa2*b2;  b2 = alfa2*a2;	d2 = alfa2*d2; e2 = alfa2*e2;	k2 = alfa2*k2;
             	}
     			else	{b2 = 0.0;	c2 = 0.0;	d2 = 0.0;	e2 = 0.0;	k2 = 0.0;}

				// storing the reduced tile data in to temporary variables for all the tiles except last one 			
				if(t < num_tile-1)
				{
					b2 =  b1 - a[id] + b2;		c2 = c1 - c[id] + c2;
					d2 =  d1 - e[id] + d2; 		k2 = k1 - x[id] + k2;
					a2 = a1;			
				}
				else 	// updating the reduced data of the last tile in the corresponding global memory
				{	
					if(eqn_id < n)
					{
						d_b[g_id] = b1 - a[id] + b2;	d_c[g_id] = c1 - c[id] + c2;
    					d_d[g_id] = d1 - e[id] + d2;	d_y[g_id] = k1 - x[id] + k2;
    					d_a[g_id] = a1; 				d_e[g_id] = e2;
					}					   				
				}
				__syncthreads(); 	  		 				 	    		 				
    		}
    		delta*=2;
    		ddelta*=2;
    	}
    }
   
   
   
__global__ void global_cached_global_tile_splitting(double *d_a, 
													double *d_b, 
													double *d_c, 
													double *d_d, 
													double *d_e, 
													double* d_y, 
													int num_split,		// number of splitting  	
						    						int num_tile, 		// number of tiles
						    						int tile_size, 		// tile size
													int n)
	{
		int i = threadIdx.x; 	int j = blockIdx.x;
    	
    	int delta = 1;	 int ddelta = 2;	int p1, p2, q1, q2;
        double alfa1, alfa2, a1, b1, c1, d1, e1, k1, a2, b2, c2, d2, e2, k2;
    	
    	for(int lc = 0; lc < num_split; lc++)	// splitting loop	
  		{		
    		for(int t = 0; t < num_tile; t++)	// tile loop
    		{	   		
    			int eqn_id = tile_size*t + i;	// equation id	
    			int g_id = j*n + eqn_id;  		// equation global id
    				
				// Calculating ids of the required neighbors
        		p1 = g_id - delta;	  p2 = g_id - ddelta;	q1 = g_id + delta;	  q2 = g_id + ddelta;
        		
		//Reduction-I
        		if(eqn_id - ddelta >= 0 && eqn_id < n)	// if both neighbors required in Reduction-I exist in the system
        		{       
            		alfa1 = 0.0;		alfa2 = 0.0;
            			
					if((eqn_id - 3*delta) >= 0)
                		if(d_b[p2] != 0.0)	alfa1 = -d_a[p1]/d_b[p2];

            		if((eqn_id + delta) <= n-1)
                		if(d_d[g_id] != 0.0)	alfa2 = -d_e[p1]/d_d[g_id];

        			a1 = alfa1*d_a[p2];	     							b1 = alfa1*d_c[p2] + d_b[p1] + alfa2*d_a[g_id];
        			c1 = alfa1*d_d[p2] + d_c[p1] + alfa2*d_b[g_id];		d1 = alfa1*d_e[p2] + d_d[p1] + alfa2*d_c[g_id];
        			e1 = alfa2*d_e[g_id]; 								k1 = alfa1*d_y[p2] + d_y[p1] + alfa2*d_y[g_id];
            	}

        		if((eqn_id - delta >= 0 && eqn_id - ddelta < 0) && eqn_id < n) 	// if only one neighbor required in Reduction-I exist 
        		{   															// in the system and other lies outside of the system
        			alfa2 = 0.0;
					if((eqn_id + delta) <= n-1)
            	   		if( d_d[g_id] != 0.0 ) 	alfa2 = -d_e[p1]/d_d[g_id];

	           		a1 = 0.0;							b1 = 0.0;
	           		c1 = d_c[p1] + alfa2*d_b[g_id];		d1 = d_d[p1] + alfa2*d_c[g_id];
	           		e1 = alfa2*d_e[g_id]; 				k1 = d_y[p1] + alfa2*d_y[g_id];
            			
                }
                	
				__syncthreads();
				
				if(t > 0)	// updating the reduce data of the previous tile in the current tile			
				{
					int preid = g_id - tile_size;
				 	d_a[preid] = a2;		d_b[preid] = b2;		d_c[preid] = c2;		
				 	d_d[preid] = d2; 		d_e[preid] = e2;		d_y[preid] = k2;
				}
				
		//Reduction-II
        		if(eqn_id + ddelta <= n-1 )		// if both neighbors required in Reduction-II exist in the system
        		{ 	
        			alfa1 = 0.0; 		alfa2 = 0.0;

        			if((eqn_id - delta) >= 0)
            			if( d_b[g_id] != 0.0 )	alfa1 = -d_a[q1]/d_b[g_id];

        			if((eqn_id + 3*delta) <= n-1)
            			if( d_d[q2] != 0.0)   	alfa2 = -d_e[q1]/d_d[q2];

        			a2 = alfa1*d_a[g_id];							  	b2 = alfa1*d_c[g_id] + d_b[q1] + alfa2*d_a[q2];
        			c2 = alfa1*d_d[g_id] + d_c[q1] + alfa2*d_b[q2];		d2 = alfa1*d_e[g_id] + d_d[q1] + alfa2*d_c[q2];
            		e2 = alfa2*d_e[q2]; 								k2 = alfa1*d_y[g_id] + d_y[q1] + alfa2*d_y[q2];
        		}

        		if(eqn_id + delta <= n-1 && eqn_id + ddelta > n-1) // if only one neighbor required in Reduction-II exist 
				{													// in the system and other lies outside of the system
            		alfa1 = 0.0;

        			if((eqn_id - delta) >= 0)
        	   			if(d_b[g_id] != 0.0)   	alfa1 = -d_a[q1]/d_b[g_id];

        			a2 = alfa1*d_a[g_id];				b2 = alfa1*d_c[g_id] + d_b[q1];
        			c2 = alfa1*d_d[g_id] + d_c[q1];		d2 = 0.0;
        			e2 = 0.0; 							k2 = alfa1*d_y[g_id] + d_y[q1];
                }

		// Reduction-III
        		if(eqn_id - delta >= 0 && eqn_id < n)
        		{
        			if( c1 != 0.0 )   	alfa1 = d_b[g_id]/c1;
        				a1 = alfa1*a1;	b1 = alfa1*b1;	c1 = alfa1*d1; d1 = alfa1*e1; k1 = alfa1*k1;
             	}
        		else{a1 = 0.0;	b1 = 0.0;	c1 = 0.0;	d1 = 0.0;	k1 = 0.0;}

        		if(eqn_id + delta <= n-1)
        		{
       				if( c2 != 0.0 )   	alfa2 = d_d[g_id]/c2;
            			c2 = alfa2*b2;  b2 = alfa2*a2;	d2 = alfa2*d2; e2 = alfa2*e2;	k2 = alfa2*k2;
             	}
     			else{b2 = 0.0;	c2 = 0.0;	d2 = 0.0;	e2 = 0.0;	k2 = 0.0;}

				__syncthreads();
				
				// storing the reduced tile data in to temporary variables for all the tiles except last one 
				if(t < num_tile-1)
				{
					b2 = b1 - d_a[g_id] + b2;		c2 = c1 - d_c[g_id] + c2;
					d2 = d1 - d_e[g_id] + d2; 	  	k2 = k1 - d_y[g_id] + k2;
					a2 = a1;												
				}
				else
				{
    				if(eqn_id < n)		// updating the reduced data of the last tile in the corresponding global memory
    				{
    					d_b[g_id] =  b1 - d_a[g_id] + b2;		d_c[g_id] = c1 - d_c[g_id] + c2;
    					d_d[g_id] =  d1 - d_e[g_id] + d2;		d_y[g_id] = k1 - d_y[g_id] + k2;
    					d_a[g_id] = a1; 						d_e[g_id] = e2;   				
    				}   				
				} 
				__syncthreads();  		 				 	    		 				
    		}
    		delta*=2;
    		ddelta*=2;   		
    	}
    }
 
     	



__global__ void pcr_penta_splitting_2nd_stage(double *d_a, 
											  double *d_b, 
											  double *d_c, 
											  double *d_d, 
											  double *d_e, 
											  double* d_y, 
 											  int num_split,	// number of splitting required for the generated system
 											  int num_sys, 		// number of generated systems from one large system
 											  int sys_size, 	// size of the generated system
 											  int n)
	{
		int id = threadIdx.x;							// equation id in the current generated system
		int j = blockIdx.x;								// global id of current generated system
		int gen_sys_id = fmodf(j, num_sys); 			// id of the current generated system respect the paret large system	
		int sys_id = j/num_sys;							// id of the parent large system
		int g_id = sys_id*n + id*num_sys + gen_sys_id; 	// equation global id
	
// allocating memory in the shared memory for all 5 diags and right hand vector
		extern __shared__ double array [];
		 
		double *a = array;	 						double *b = (double*) &a[sys_size];
    	double *c = (double*) &b[sys_size];	 		double *d = (double*) &c[sys_size];
    	double *e = (double*) &d[sys_size];	 		double *x = (double*) &e[sys_size];
    
// copying the data of current generated system from the global to shared memory
		a[id] = d_a[g_id];		b[id] = d_b[g_id]; 		c[id] = d_c[g_id]; 
		d[id] = d_d[g_id]; 		e[id] = d_e[g_id]; 		x[id] = d_y[g_id];
	
    	__syncthreads();
    
// executing all PCR steps by for loop 
		int delta = 1;		int p1, p2, q1, q2; 	
		double alfa1, alfa2, a1, b1, c1, d1, e1, k1, a2, b2, c2, d2, e2, k2;
	
  		for(int l = 0; l< num_split-1; l++)
		{
			// Calculating ids of the required neighbors
			p1 = id - delta;	  p2 = id - 2*delta;	q1 = id + delta;	  q2 = id + 2*delta;
		
    	//Reduction-I       
			if(p2 >= 0)	// if both neighbors required in Reduction-I exist in the system
        	{       
            	alfa1 = 0.0;		alfa2 = 0.0;
            	if((id - 3*delta) >= 0)
             	   if( b[p2] != 0.0 )	alfa1 = -a[p1]/b[p2];
            	if(q1 <= sys_size-1)
                 	if( d[id] != 0.0 )	alfa2 = -e[p1]/d[id];

            	a1 = alfa1*a[p2];	     					b1 = alfa1*c[p2] + b[p1] + alfa2*a[id];
            	c1 = alfa1*d[p2] + c[p1] + alfa2*b[id];		d1 = alfa1*e[p2] + d[p1] + alfa2*c[id];
            	e1 = alfa2*e[id]; 							k1 = alfa1*x[p2] + x[p1] + alfa2*x[id];
       	 	}

        	if(p1 >= 0 && p2 < 0)	// if only one neighbor required in Reduction-I exist 
        	{   					// in the system and other lies outside of the system
         		alfa2 = 0.0;
            	if(q1 <=sys_size-1)
              	 	if( d[id] != 0.0 ) 	alfa2 = -e[p1]/d[id]; 

           		a1 = 0.0;						b1 = 0.0;
            	c1 = c[p1] + alfa2*b[id];		d1 = d[p1] + alfa2*c[id];
            	e1 = alfa2*e[id]; 				k1 = x[p1] + alfa2*x[id];
         	}

		//Reduction-II
        	if(q2 <= sys_size-1 )	// if both neighbors required in Reduction-II exist in the system
        	{ 	
        		alfa1 = 0.0; 		alfa2 = 0.0;

            	if(p1 >= 0)
            		if( b[id] != 0.0 )		alfa1 = -a[q1]/b[id];
             	if((id + 3*delta) <= sys_size-1)
             		if( d[q2] != 0.0)   	alfa2 = -e[q1]/d[q2];

             	a2 = alfa1*a[id];							b2 = alfa1*c[id] + b[q1] + alfa2*a[q2];
             	c2 = alfa1*d[id] + c[q1] + alfa2*b[q2];	d2 = alfa1*e[id] + d[q1] + alfa2*c[q2];
             	e2 = alfa2*e[q2]; 							k2 = alfa1*x[id] + x[q1] + alfa2*x[q2];
          	}
          
			if(q1 <= sys_size-1 && q2 > sys_size-1) 	// if only one neighbor required in Reduction-II exist 
			{										// in the system and other lies outside of the system
            	alfa1 = 0.0;
				if(p1 >= 0)
            		if(b[id] != 0.0)   	alfa1 = -a[q1]/b[id];

            	a2 = alfa1*a[id];				b2 = alfa1*c[id] + b[q1];
            	c2 = alfa1*d[id] + c[q1];		d2 = 0.0;
            	e2 = 0.0; 						k2 = alfa1*x[id] + x[q1];
         	}

		// Reduction-III
         	if(p1 >= 0)
         	{
          		if( c1 != 0.0 )   	alfa1 = b[id]/c1;
            		a1 = alfa1*a1;	b1 = alfa1*b1;	c1 = alfa1*d1; d1 = alfa1*e1; k1 = alfa1*k1;
	     	}
         	else	{a1 = 0.0;	b1 = 0.0;	c1 = 0.0;	d1 = 0.0;	k1 = 0.0;}

         	if(q1 <= sys_size-1)
         	{
	     		if( c2 != 0.0 )   	alfa2 = d[id]/c2;
            	c2 = alfa2*b2;  b2 = alfa2*a2;	d2 = alfa2*d2; e2 = alfa2*e2;	k2 = alfa2*k2;
	      	}
	      	else	{b2 = 0.0;	c2 = 0.0;	d2 = 0.0;	e2 = 0.0;	k2 = 0.0;}
	      
			// waiting for every threads to complete Reduction-I and -II 
		  	__syncthreads();
		  
		  	delta*=2;
		  	num_sys*=2;
	      	b[id] = b1 - a[id] + b2;		c[id] = c1 - c[id] + c2;
	      	d[id] = d1 - e[id] + d2; 		x[id] = k1 - x[id] + k2;
	      	a[id] = a1; 					e[id] = e2;
    		// waiting for every threads to complete Reduction-III 
    	  	__syncthreads();  	   	 
    	}
// for the last loop system size is less than or equal to 2, thus efficient to solve them manually
  		if(id < sys_size-delta)
		{
			int iid = id + delta;
			double deno = c[id]*c[iid]-b[iid]*d[id];
			d_y[g_id] = (c[iid]*x[id]-d[id]*x[iid])/deno;
			d_y[g_id + num_sys] = (c[id]*x[iid]-b[iid]*x[id])/deno;
		}
		if(id >= sys_size-delta && id < delta)
  	   		d_y[g_id] = x[id]/c[id];
	}


