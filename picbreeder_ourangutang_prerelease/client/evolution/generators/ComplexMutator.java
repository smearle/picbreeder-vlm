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

import client.evolution.*;

public final class ComplexMutator extends AbstractMutator {
	private final Mutator first, second;
	
	public ComplexMutator(Mutator a, Mutator b) {
		first = a;
		second = b;
	}
	
	public void mutate(Genome offspring) {
		first.mutate(offspring);
		second.mutate(offspring);
	}
}
