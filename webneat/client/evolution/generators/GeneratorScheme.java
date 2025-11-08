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

package client.evolution.generators;

import client.evolution.Generator;
import client.evolution.Genome;

import java.util.Collection;
import java.util.Map;

public class GeneratorScheme implements Generator {
	private Generator activeScheme = null;
	private Map <String, Generator> schemes = new java.util.TreeMap <String, Generator> ();

	public Genome generate(Collection<Genome> parents) {
		return activeScheme.generate(parents);
	}
	
	public int minimumParents() {
		return 1;
	}
	
	public void addScheme(String name, Generator generator) {
		if(activeScheme == null)
			activeScheme = generator;
		
		schemes.put(name, generator);
	}
	
	public void pickScheme(String name) {
		activeScheme = schemes.get(name);
	}
}
