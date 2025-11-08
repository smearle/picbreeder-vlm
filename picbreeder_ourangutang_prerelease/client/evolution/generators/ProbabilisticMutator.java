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

import client.evolution.Genome;
import client.utilities.Random;

final class ProbabilisticMutator extends AbstractMutator {
	private final double probability;
	private final Mutator mutator;
	
	protected ProbabilisticMutator(Mutator mutator, double probability) {
		this.mutator = mutator;
		this.probability = probability;
	}
	
	public final void mutate(Genome g) {
		if(Random.instance().nextBoolean(probability))
			mutator.mutate(g);
	}
}
