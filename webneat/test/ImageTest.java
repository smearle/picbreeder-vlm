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

import java.io.File;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;

import client.evolution.*;
import client.evolution.impl.DefaultGeneticFactory;
import client.math.SampledBoundedFunction;
import client.cppn.*;
import client.math.*;

public class ImageTest {
	public static void main(String [] args) throws Exception {
		/*if(args.length == 0) {
			System.out.println("You must specify files to echo.");
			return;
		}
		
		GeneticFactoryInstance.set(new client.evolution.impl.DefaultGeneticFactory());
		*/
		for(String x : new String[]{"file1.out"})
			write(x);
	}
	
	public static void write(String file) throws Exception {
		/*Series s = GeneticFactoryInstance.get().createInvalidSeries();
		client.utilities.XML.load(s, file);
		
		Genome g = s.findGenome(1);
		
		if(g == null) {
			System.out.println("Invalid genome id number.");
			System.exit(-1);
		}
		
		Network network = CPPNFactoryInstance.get().createNetwork(g);
		test.ImagePhenotype image = new test.ImagePhenotype();
		*/
	/*	Function f = new SampledBoundedFunction(new UnipolarToBipolar(new Sigmoid()), -25, 25, 1e-3);
		final int trials = 100;
		final int runs = 10000000;
		for(int i = 0; i < trials; i++){
			long start = System.currentTimeMillis();
			for(int j = 0; j < runs; j++)
				f.valueAt(Math.random()*10.0-5.0);
			long end = System.currentTimeMillis();
			System.out.println(end - start);
		}*/
		//image.save(file + ".jpg");
	}
}
