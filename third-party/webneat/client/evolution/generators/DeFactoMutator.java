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

public final class DeFactoMutator extends SequenceMutator {
	public DeFactoMutator() {
		final double linkProbability = client.ParameterTableInstance.get().getDouble("evolution", "add link rate");
		final double nodeProbability = client.ParameterTableInstance.get().getDouble("evolution", "add node rate");
		
		addMutator(new ProbabilisticMutator(new AddAcyclicLink(), linkProbability));
		addMutator(new ProbabilisticMutator(new AddNodes(), nodeProbability));
		addMutator(new MutateLinks());
	}
}
