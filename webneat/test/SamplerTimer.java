/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package test;

import client.math.*;
import java.util.*;

public class SamplerTimer {
	
	static final int count = 100000000;

	public static void main(String []args) {
	/*	Function sigmoid = new UnipolarToBipolar(new Sigmoid());
		Function sampledSigmoid = new SampledBoundedFunction(sigmoid, -25, 25, 1e-3);
		Function psSigmoid = new PowerSeriesSigmoid();
		
		testIt(sigmoid);
		testIt(sampledSigmoid);
		testIt(psSigmoid);
		

		Function gaussian = new UnipolarToBipolar(new Gaussian());
		Function sampledGaussian = new SampledBoundedFunction(gaussian, -25, 25, 1e-3);
		Function psGaussian = new PowerSeriesGaussian();
		
		testIt(gaussian);
		testIt(sampledGaussian);
		testIt(psGaussian);*/
	}
	
	public static void testIt(Function f) {
		Random r = new Random(0);
		long s = System.currentTimeMillis();
		for(int i = 0; i < count; i++)
			f.valueAt(r.nextDouble());
		long t = System.currentTimeMillis();
		
		System.out.println("Time: " + (t - s) * 1000.0 / count + " microseconds");
	}
}
