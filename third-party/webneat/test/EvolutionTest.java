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

import client.cppn.CPPNFactory;
import client.cppn.impl.DefaultCPPNFactory;
import client.evolution.*;
import client.evolution.impl.DefaultGeneticFactory;
import client.utilities.*;

public class EvolutionTest {
	public static void main(String []args) throws Exception {
		Series s = GeneticFactoryInstance.get().createRootSeries();
		
		for(int i = 0; i < 50; i++) {
			s.getIndividualFromCurrentGeneration(0).select();
			s.spawn();
		}
		
		XML.store(s, "series.xml");
		
	}
}
