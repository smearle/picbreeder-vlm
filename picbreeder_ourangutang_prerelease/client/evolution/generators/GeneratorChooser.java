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
import java.util.*;

/**
 * The GeneratorChooser is used to pick randomly from many different
 * generation schemes. The desired generators are added to the
 * chooser, the chooser is locked, and then the chooser will
 * pick from the given generators each time the generate
 * method is invoked.
 * 
 * @author Nick
 */

public final class GeneratorChooser implements Generator {
	private final List <Generator> generators;
	private int minimumParents;
	private boolean locked;

	/**
	 * Constructs a chooser by adding the generators and
	 * locking the chooser. This chooser is then ready to
	 * generate genomes.
	 * 
	 * @param generators The set of generators
	 */
	
	public GeneratorChooser(Collection <Generator> generators) {
		this();
		
		this.generators.addAll(generators);
		lock();
	}
	
	/**
	 * Constructs a chooser that is expected to be built
	 * later and locked before use.
	 */
	public GeneratorChooser() {
		locked = false;
		generators = new LinkedList <Generator> ();
	}
	
	public Genome generate(Collection <Genome> parents) {
		assert(locked);
		assert(minimumParents <= parents.size());
		
		ArrayList <Generator> possible = new ArrayList <Generator> ();
		for(Generator g : generators)
			if(g.minimumParents() <= parents.size())
				possible.add(g);
		
		Generator chosenOne = possible.get(client.utilities.Random.instance().nextInt(possible.size()));
		return chosenOne.generate(parents);
	}
	
	public int minimumParents() {
		assert(locked);
		return minimumParents;
	}
	
	/**
	 * Adds a generator (provided this is not locked) to
	 * the set of possible generators. The same generator
	 * may be added multiple times to increase its chance
	 * of use.
	 * 
	 * @param generator The generator to add
	 */
	public void addGenerator(Generator generator) {
		assert(!locked);
		generators.add(generator);
	}
	
	/**
	 * Locks this generator so that it may be used. Invoking
	 * this method is undo-able and will disallow addition
	 * of new generators.
	 */
	public void lock() {
		int min = Integer.MAX_VALUE;
		for(Generator g : generators)
			min = Math.min(min, g.minimumParents());
		
		locked = true;
	}
}
