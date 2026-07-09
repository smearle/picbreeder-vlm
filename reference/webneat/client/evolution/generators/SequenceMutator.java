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

import java.util.LinkedList;
import client.evolution.Genome;

public class SequenceMutator extends AbstractMutator {
	private LinkedList <Mutator> mutationSequence;
	
	protected SequenceMutator() {
		mutationSequence = new LinkedList <Mutator> ();
	}

	public final void mutate(Genome g) {
		for(Mutator m : mutationSequence)
			m.mutate(g);
	}
	
	protected final void addMutator(Mutator m) {
		mutationSequence.add(m);
	}
}
